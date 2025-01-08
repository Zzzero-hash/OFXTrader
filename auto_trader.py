import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import forex_env
import ta  # Technical Analysis library
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
import optuna
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple
import multiprocessing
import logging
import gym
import subprocess
import sys

logging.basicConfig(level=logging.INFO)

# Enforced CUDA usage
if not torch.cuda.is_available():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "stable-baselines3[cuda]"])
    except subprocess.CalledProcessError as e:
        raise EnvironmentError("CUDA is not available and failed to install stable-baselines3 with CUDA support.") from e

# Verify CUDA availability using nvidia-smi
try:
    cuda_available = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).returncode == 0
except FileNotFoundError:
    cuda_available = False

# Add dynamic device assignment
device = 'cuda' if cuda_available else 'cpu'
logging.info(f"Using device: {device}")

@lru_cache(maxsize=32)
def fetch_oanda_candles_chunk(instrument, from_time, to_time, granularity, access_token):
    """Fetch up to `count` candles in one request."""
    client = oandapyV20.API(access_token=access_token)
    params = {
        "from": pd.Timestamp(from_time).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "to": pd.Timestamp(to_time).strftime('%Y-%m-%dT%H:%M:%SZ'),      
        "granularity": granularity
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    resp = client.request(r)
    
    # Parse into DataFrame
    data = resp["candles"]
    df = pd.DataFrame([{
        'time': candle['time'],
        'open': float(candle['mid']['o']),
        'high': float(candle['mid']['h']),
        'low': float(candle['mid']['l']),
        'close': float(candle['mid']['c']),
        'volume': int(candle['volume'])
    } for candle in data])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    return df

def fetch_oanda_candles_range(instrument, start_date, end_date, granularity, access_token):
    current_start = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Calculate chunks for parallel processing
    chunks = []
    while current_start < end_date:
        to_dt = min(current_start + pd.Timedelta(minutes=5000), end_date)
        chunks.append((current_start, to_dt))
        current_start = to_dt

    # Parallel fetch chunks
    max_workers = min(len(chunks), multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fetch_oanda_candles_chunk,
                instrument,
                chunk[0],
                chunk[1],
                granularity,
                access_token
            ) for chunk in chunks
        ]
        results = [future.result() for future in futures]
    
    return pd.concat(results, ignore_index=True)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    
    # Vectorized calculations
    df['rsi_14'] = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    df['dx_14'] = ta.trend.ADXIndicator(high=high_prices, low=low_prices, close=close_prices, window=14).adx()
    
    # Optimize Bollinger Bands calculation
    bb = ta.volatility.BollingerBands(close=close_prices, window=14)
    df['bb_ub'] = bb.bollinger_hband()
    df['bb_lb'] = bb.bollinger_lband()
    
    # Use inplace operations
    df.dropna(inplace=True)
    return df

def train_model(train_data, timesteps=50000, stop_loss_percent=0.02, use_optuna=False, n_trials=10):
    if use_optuna:
        return hyperparam_tuning(train_data, n_trials=n_trials)
    
    if device == 'cuda':
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
    
    env_train = forex_env.OandaForexTradingEnv(
        data_sequences=train_data,
        stop_loss_percent=stop_loss_percent
    )
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", device=device)  # Use dynamic device
    trained_model = agent.train_model(model=model, tb_log_name='ppo', total_timesteps=timesteps)
    return trained_model

def calculate_sharpe_ratio(rewards: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio with numerical stability.

    Parameters:
    rewards (List[float]): A list of reward values.
    risk_free_rate (float): The risk-free rate of return. Default is 0.0.

    Returns:
    float: The calculated Sharpe Ratio.
    """
    if not rewards or len(rewards) < 2:
        return 0.0
    
    returns = np.array(rewards)
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    return mean_excess / std_excess if std_excess != 0 else 0.0

def calculate_sortino_ratio(rewards: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino Ratio of the rewards.

    Parameters:
    rewards (List[float]): A list of reward values.
    risk_free_rate (float): The risk-free rate of return. Default is 0.0.

    Returns:
    float: The calculated Sortino Ratio.
    """
    returns = np.array(rewards)
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    mean_excess = np.mean(excess_returns)
    std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    return mean_excess / std_downside if std_downside != 0 else 0.0

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Calculate the Maximum Drawdown of the portfolio.

    Parameters:
    portfolio_values (List[float]): A list of portfolio values.

    Returns:
    float: The calculated Maximum Drawdown.
    """
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

def calculate_profit_factor(profit: float, loss: float) -> float:
    """
    Calculate the Profit Factor with limits.

    Parameters:
    profit (float): The total profit value.
    loss (float): The total loss value.

    Returns:
    float: The calculated Profit Factor.
    """
    if abs(loss) < 1e-8:
        return 1.0 if profit == 0 else 10.0  # More reasonable cap
    ratio = abs(profit / loss) if loss != 0 else 1.0
    return min(ratio, 10.0)  # Cap at 10.0

def check_numerical_stability(metrics: Dict[str, float]) -> bool:
    """
    Check for numerical instability in metrics.

    Parameters:
    metrics (Dict[str, float]): A dictionary of metric names and values.

    Returns:
    bool: True if metrics are stable, False otherwise.
    """
    for key, value in metrics.items():
        if not np.isfinite(value) or abs(value) > 1e6:
            logging.warning(f"Numerical instability detected in {key}: {value}")
            return False
    return True

def test_model(test_data: pd.DataFrame, model: PPO) -> Tuple[float, Dict[str, float]]:
    """
    Test the model on test data and calculate performance metrics.

    Parameters:
    test_data (pd.DataFrame): The test data containing historical price and technical indicators.
    model (PPO): The trained PPO model.

    Returns:
    float: The total reward.
    Dict[str, float]: A dictionary of performance metrics.
    """
    env_test = forex_env.OandaForexTradingEnv(data_sequences=test_data)
    obs, info = env_test.reset()
    done = False
    total_reward = 0
    rewards = []
    portfolio_values = [env_test.capital]
    total_profit = 0
    total_loss = 0

    while not done:
        action, _states = model.predict(np.array(obs))
        obs, reward, done, truncated, info = env_test.step(action)
        total_reward += reward
        rewards.append(reward)
        portfolio_values.append(env_test.capital)
        if reward > 0:
            total_profit += reward
        else:
            total_loss += reward

    sharpe = calculate_sharpe_ratio(rewards)
    sortino = calculate_sortino_ratio(rewards)
    max_dd = calculate_max_drawdown(portfolio_values)
    profit_factor = calculate_profit_factor(total_profit, total_loss)
    
    metrics = {
        'Total Reward': total_reward,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Maximum Drawdown': max_dd,
        'Profit Factor': profit_factor
    }
    
    # Add numerical stability checks
    if not check_numerical_stability(metrics):
        logging.warning("Numerical instability detected, returning safe values")
        metrics = {
            'Total Reward': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Maximum Drawdown': 1.0,
            'Profit Factor': 1.0
        }
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return total_reward, metrics

def check_performance(metrics: Dict[str, float], threshold: float = 0.5) -> bool:
    """
    Enhanced performance checking.

    Parameters:
    metrics (Dict[str, float]): A dictionary of performance metrics.
    threshold (float): The threshold for performance metrics. Default is 0.5.

    Returns:
    bool: True if performance meets the criteria, False otherwise.
    """
    sharpe_ratio = metrics.get('Sharpe Ratio', 0.0)
    sortino_ratio = metrics.get('Sortino Ratio', 0.0)
    max_drawdown = metrics.get('Maximum Drawdown', 1.0)
    profit_factor = metrics.get('Profit Factor', 0.0)
    
    # More comprehensive evaluation
    conditions = [
        sharpe_ratio >= threshold,
        sortino_ratio >= threshold,
        max_drawdown <= 0.3,  # Maximum 30% drawdown
        profit_factor >= 1.2   # At least 1.2 profit factor
    ]
    
    return sum(conditions) >= 3  # At least 3 conditions must be met

def auto_retrain(instrument: str, granularity: str, start_date: str, end_date: str, access_token: str, timesteps: int = 50000, use_optuna: bool = False, n_trials: int = 10):
    """
    Retrain model by fetching new data and using FinRL's DRLAgent.

    Parameters:
    instrument (str): The trading instrument (e.g., 'EUR_USD').
    granularity (str): The granularity of the data (e.g., 'H1').
    start_date (str): The start date for fetching data in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end_date (str): The end date for fetching data in ISO 8601 format (e.g., '2024-12-31T00:00:00Z').
    access_token (str): The access token for OANDA API.
    timesteps (int, optional): The number of timesteps for training. Default is 50000.
    use_optuna (bool, optional): Whether to use Optuna for hyperparameter tuning. Default is False.
    n_trials (int, optional): The number of trials for Optuna hyperparameter tuning. Default is 10.

    Returns:
    model: The retrained model.
    """
    new_data = fetch_oanda_candles_range(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        access_token=access_token
    )
    new_data = add_technical_indicators(new_data)
    train_data = new_data[:int(0.8 * len(new_data))]
    new_model = train_model(train_data, timesteps=timesteps, use_optuna=use_optuna, n_trials=n_trials)
    return new_model

def run_bot_with_monitoring(instrument: str, granularity: str, start_date: str, end_date: str, access_token: str, use_optuna: bool = False, n_trials: int = 10):
    """
    Main loop that executes trades and checks performance regularly.

    Parameters:
    instrument (str): The trading instrument (e.g., 'EUR_USD').
    granularity (str): The granularity of the data (e.g., 'H1' for hourly data).
    start_date (str): The start date for fetching data in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').
    end_date (str): The end date for fetching data in ISO 8601 format (e.g., '2024-12-31T00:00:00Z').
    access_token (str): The access token for OANDA API.
    use_optuna (bool): Whether to use Optuna for hyperparameter tuning. Default is False.
    n_trials (int): The number of trials for Optuna hyperparameter tuning. Default is 10.
    """
    df = fetch_oanda_candles_range(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        access_token=access_token,
        granularity=granularity
    )
    df = add_technical_indicators(df)
    train_data = df[:int(0.8 * len(df))]
    test_data = df[int(0.8 * len(df)):]
    model = train_model(train_data, use_optuna=use_optuna, n_trials=n_trials)
    while True:
        test_reward, metrics = test_model(test_data, model)
        break  # Temporary break for testing
        if not check_performance(metrics, threshold=1.0):  # Pass actual metrics
            logging.info("Performance below threshold, retraining...")
            model = auto_retrain(instrument, granularity, start_date, end_date, access_token, use_optuna=use_optuna, n_trials=n_trials)
        # Insert actual trade execution calls here
        # Optionally, log metrics
        logging.info(f"Test Reward: {test_reward}, Metrics: {metrics}")

def hyperparam_tuning(train_data: pd.DataFrame, n_trials: int = 10) -> PPO:
    """
    Perform hyperparameter tuning using Optuna.

    Parameters:
    train_data (pd.DataFrame): The training data containing historical price and technical indicators.
    n_trials (int): The number of trials for Optuna hyperparameter tuning. Default is 10.

    Returns:
    PPO: The trained PPO model with the best hyperparameters.
    """
    def objective(trial):
        # Narrow down parameter ranges
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
        n_steps = trial.suggest_int('n_steps', 256, 512)
        gamma = trial.suggest_float('gamma', 0.98, 0.995)
        ent_coef = trial.suggest_float('ent_coef', 1e-5, 1e-4)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        
        # Additional PPO parameters
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 0.7)
        vf_coef = trial.suggest_float('vf_coef', 0.4, 0.6)
        
        env_train = forex_env.OandaForexTradingEnv(
            data_sequences=train_data
        )
        
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            batch_size=batch_size,
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            policy_kwargs={"net_arch": [128, 128]},  # Deeper network
            device=device
        )

        model.learn(total_timesteps=20000)
        train_reward, _ = test_model(train_data, model)
        return train_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logging.info("Best hyperparams: %s", best_params)
    
    # Retrain final model with best parameters
    final_env = forex_env.OandaForexTradingEnv(
        data_sequences=train_data 
    )
    final_model = PPO(
        "MlpPolicy",
        final_env,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        batch_size=best_params['batch_size'],
        clip_range=best_params['clip_range'],
        max_grad_norm=best_params['max_grad_norm'],
        vf_coef=best_params['vf_coef'],
        policy_kwargs={"net_arch": [128, 128]},  # Deeper network
        device=device  # Use dynamic device
    )
    final_model.learn(total_timesteps=50000)
    return final_model

# Example usage
if __name__ == "__main__":
    instrument = 'EUR_USD'
    granularity = 'M5'
    start_date = '2020-01-01T00:00:00Z'
    end_date = '2024-12-31T00:00:00Z'
    access_token = 'a15df916d468a21855b25932c59b6947-de38c8e63794cf1d040c170d1ca6df24'

    run_bot_with_monitoring(instrument, granularity, start_date, end_date, access_token, use_optuna=True, n_trials=10)