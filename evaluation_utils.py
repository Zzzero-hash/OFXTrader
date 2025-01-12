import logging
import numpy as np
from typing import Dict, List, Tuple
from forex_env import create_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

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

def test_model(test_data, model: PPO, n_eval_episodes: int = 5) -> Tuple[float, Dict[str, float]]:
    env_test = create_env(
        df=test_data,
        window_size=10,
        frame_bound=(10, len(test_data)),
        unit_side='right'
    )

    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    all_rewards = []
    all_portfolio_values = []

    for episode in range(n_eval_episodes):
        obs = env_test.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            all_rewards.append(reward)
            all_portfolio_values.append(env_test.capital)

    # Compute additional metrics
    sharpe = calculate_sharpe_ratio(all_rewards)
    sortino = calculate_sortino_ratio(all_rewards)
    max_dd = calculate_max_drawdown(all_portfolio_values)
    profit_factor = calculate_profit_factor(
        profit=env_test.trade_metrics['total_profit'],
        loss=abs(env_test.trade_metrics['total_profit'] - env_test.capital)
    )

    # Calculate profit and loss percentages
    initial_capital = env_test.initial_capital
    profit_percent = (env_test.trade_metrics['total_profit'] / initial_capital) * 100
    loss_percent = (abs(env_test.trade_metrics['total_profit'] - env_test.capital) / initial_capital) * 100

    metrics = {
        'Mean Reward': mean_reward,
        'Std Reward': std_reward,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Maximum Drawdown': max_dd,
        'Profit Factor': profit_factor,
        'profit_percent': profit_percent,
        'loss_percent': loss_percent
    }

    # Check for numerical stability
    if not check_numerical_stability(metrics):
        logging.warning("Numerical instability detected, returning safe values")
        metrics = {
            'Mean Reward': 0.0,
            'Std Reward': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Maximum Drawdown': 1.0,
            'Profit Factor': 1.0,
            'profit_percent': 0.0,
            'loss_percent': 0.0
        }

    # Print the results
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return mean_reward, metrics

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

class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        # Access training info and log custom metrics here if needed
        # Example: metrics can come from self.model.rollout_buffer or env
        return True