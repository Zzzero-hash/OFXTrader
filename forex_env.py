import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data_handler import DataHandler
from gymnasium.envs.registration import register
import gymnasium.utils.seeding as seeding

class ForexEnv(gym.Env):
    def __init__(self, instrument="EUR_USD", start_date="2022-01-01", end_date="2023-01-01", 
                 granularity="M1", initial_balance=1000, leverage=10, window_size=14, 
                 spread_pips=0.0001, render_frequency=1000):
        super(ForexEnv, self).__init__()
        self.data_handler = DataHandler()
        self.data, self.feature_names = self.data_handler.get_data(instrument, start_date, end_date, granularity, window_size)
        if self.data.size == 0:
            raise ValueError("No data fetched. Check your DataHandler configuration.")
        self.n_timesteps, self.n_features = self.data.shape
        self.close_idx = self.feature_names.index('close')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: close, 1: long, 2: short
        self.data = self.data.astype(np.float32)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.spread = spread_pips
        self.render_frequency = render_frequency
        self.max_steps = 500  # Reduced for faster feedback
        self.position = 0
        self.entry_price = None
        self.entry_step = 0
        self.current_step = 0
        self.total_profit = 0
        self.done = False
        self.truncated = False
        self.spec = gym.envs.registration.EnvSpec(id='forex-v0', entry_point='forex_env:ForexEnv', max_episode_steps=self.max_steps)

    def reset(self, *, seed=None, options=None):
        self.seed(seed)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.entry_step = 0
        self.last_profit = 0
        self.current_step = 0
        self.total_profit = 0
        self.done = False
        self.truncated = False
        observation = self._next_observation().astype(np.float32)
        print(f"[RESET] Observation shape: {observation.shape}, dtype: {observation.dtype}")
        return observation, {}

    def _next_observation(self):
        if self.current_step >= self.n_timesteps:
            self.done = True
            return np.zeros((self.n_features,), dtype=np.float32)
        return self.data[self.current_step].astype(np.float32)

    def _calculate_profit(self, exit_price):
        if self.position == 0:
            return 0
        if self.position == 1:  # Long
            profit = (exit_price - self.entry_price) * self.leverage
        else:  # Short
            profit = (self.entry_price - exit_price) * self.leverage
        self.balance += profit
        self.total_profit += profit
        self.position = 0
        self.entry_price = None
        return profit

    def _take_action(self, action):
        current_price = self.data[self.current_step, self.close_idx]
        self.last_profit = 0
        if action == 0:
            if self.position != 0:
                self.last_profit = self._calculate_profit(current_price)
        else:
            if action == 1:  # Long
                if self.position == -1:
                    self.last_profit = self._calculate_profit(current_price)
                self.entry_price = current_price + self.spread
                self.position = 1
                self.entry_step = self.current_step
            elif action == 2:  # Short
                if self.position == 1:
                    self.last_profit = self._calculate_profit(current_price)
                self.entry_price = current_price - self.spread
                self.position = -1
                self.entry_step = self.current_step

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        time_penalty = 0.02 * (self.current_step - self.entry_step) if self.position != 0 else 0.0
        drawdown_penalty = -50.0 if self.balance <= 0.4 * self.initial_balance else 0.0
        commission = 0.0001 * abs(self.last_profit)
        reward = self.last_profit - commission - time_penalty + drawdown_penalty
        if not np.isfinite(reward):
            logger.error(f"Invalid reward: last_profit={self.last_profit}, commission={commission}, time_penalty={time_penalty}, drawdown_penalty={drawdown_penalty}")
            reward = -1.0
        reward_normalized = np.clip(reward / self.initial_balance, -1.0, 1.0)
        self.done = self.current_step >= self.max_steps
        self.truncated = self.balance <= 0.5 * self.initial_balance
        if self.render_frequency > 0 and self.current_step % self.render_frequency == 0:
            self.render()
        return self._next_observation(), reward_normalized, self.done, self.truncated, {'balance': self.balance, 'position': self.position}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Total Profit + (Unrealized P/L): {self.total_profit:.2f}')
        print('---------------------------------')

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Register the environment
register(id='forex-v0', entry_point='forex_env:ForexEnv')