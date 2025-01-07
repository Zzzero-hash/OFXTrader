import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import datetime
from stable_baselines3 import PPO
import forex_env
import ta  # Technical Analysis library
from finrl.config_tickers import FX_TICKER
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
import optuna
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List
import multiprocessing

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
    with ThreadPoolExecutor(max_workers=min(len(chunks), multiprocessing.cpu_count() * 2)) as executor:
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
    # Pre-calculate common values
    close_prices = pd.Series(df['close'].values)
    high_prices = pd.Series(df['high'].values)
    low_prices = pd.Series(df['low'].values)
    
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
    
    if torch.cuda.is_available():
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    env_train = forex_env.OandaForexTradingEnv(
        data_sequences=train_data,
        stop_loss_percent=stop_loss_percent
    )
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", device='cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = agent.train_model(model=model, tb_log_name='ppo', total_timesteps=timesteps)
    return trained_model

def test_model(test_data, model):
    env_test = forex_env.OandaForexTradingEnv(data_sequences=test_data)
    obs, info = env_test.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(np.array(obs))
        obs, reward, done, truncated, info = env_test.step(action)
        total_reward += reward
    return total_reward

def check_performance(env, threshold=0.9):
    """
    Evaluate model performance. If performance is below threshold, return False.
    Replace this logic with your own performance metrics.
    """
    performance_score = 1.0  # Stub for actual evaluation
    return performance_score >= threshold

def auto_retrain(instrument, granularity, start_date, end_date, access_token, timesteps=50000, use_optuna=False, n_trials=10):
    """
    Retrain model by fetching new data and using FinRL's DRLAgent.
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

def run_bot_with_monitoring(instrument, granularity, start_date, end_date, access_token, use_optuna=False, n_trials=10):
    """
    Main loop that executes trades and checks performance regularly.
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
        test_reward = test_model(test_data, model)
        if not check_performance(env=None):  # Pass appropriate env or data
            print("Performance below threshold, retraining...")
            model = auto_retrain(instrument, granularity, start_date, end_date, access_token, use_optuna=use_optuna, n_trials=n_trials)
        # Insert actual trade execution calls here

def hyperparam_tuning(train_data, n_trials=10):
    """
    Hyperparameter optimization using Optuna.
    """
    def objective(trial):
        # Example search space, adjust as desired
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        
        env_train = forex_env.OandaForexTradingEnv(
            data_sequences=train_data,
            stop_loss_percent=0.02
        )
        agent = DRLAgent(env=env_train)
        
        # Initialize PPO with hyperparameters
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            batch_size=batch_size,
            policy_kwargs={"net_arch": [64, 64]},
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train model
        model.learn(total_timesteps=20000)
        reward = test_model(train_data, model)
        return -reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print("Best hyperparams:", best_params)
    
    # Retrain final model with best parameters
    final_env = forex_env.OandaForexTradingEnv(data_sequences=train_data)
    final_model = PPO(
        "MlpPolicy",
        final_env,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        batch_size=best_params['batch_size'],
        policy_kwargs={"net_arch": [64, 64]},
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    final_model.learn(total_timesteps=50000)
    return final_model

# Example usage
instrument = 'EUR_USD'
granularity = 'M5'
start_date = '2017-01-01T00:00:00Z'
end_date = '2024-12-31T00:00:00Z'
access_token = 'a15df916d468a21855b25932c59b6947-de38c8e63794cf1d040c170d1ca6df24'

run_bot_with_monitoring(instrument, granularity, start_date, end_date, access_token, use_optuna=True, n_trials=10)