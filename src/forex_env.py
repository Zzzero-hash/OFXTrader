import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from src.data_handler import DataHandler

class ForexEnv(gym.Env):
    def __init__(self, instrument, start_date, end_date, granularity, initial_balance=1000, leverage=50, window_size=10, spread_pips=0.0001):
        super(ForexEnv, self).__init__()
        self.data_handler = DataHandler()
        self.data, self.feature_names = self.data_handler.get_data(instrument, start_date, end_date, granularity, window_size)
        self.n_windows, self.window_size, self.n_features = self.data.shape
        self.close_idx = self.feature_names.index('close')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, self.n_features))
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.window_size = window_size
        self.position = 0
        self.last_trade_price = 0
        self.profit = 0
        self.spread = spread_pips
        self.entry_price = None
        self.done = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        # Optional: If needed, call the parent class reset
        # super().reset(seed=seed, options=options)
        assert len(self.data) >= self.window_size, f"Need at least {self.window_size} windows, got {len(self.data)}"
        self.balance = self.initial_balance
        self.position = 0
        self.profit = 0
        self.entry_price = 0
        self.done = False
        self.truncated = False
        self.current_step = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        if self.current_step >= self.n_windows:
            self.done = True
            return np.zeros((self.window_size, self.n_features))
        return self.data[self.current_step]
    
    def _take_action(self, action):
        current_price = self.data[self.current_step, -1, self.close_idx]
        if action == 0:
            if self.position != 0:
                self._calculate_profit(current_price)
            self.position = 0
        elif action == 1:
            if self.position != 1:
                if self.position == -1:
                    self._calculate_profit(current_price)
                self.entry_price = current_price + self.spread
                self.position = 1
        elif action == 2:
            if self.position != -1:
                if self.position == 1:
                    self._calculate_profit(current_price)
                self.entry_price = current_price - self.spread
                self.position = -1
        else:
            raise ValueError(f"Invalid action: {action}")
        
    def _calculate_profit(self, exit_price):
        if self.position == 1:
            profit = (exit_price - self.entry_price) * self.leverage
        elif self.position == -1:
            profit = (self.entry_price - exit_price) * self.leverage
        else:
            return self.balance + profit
        
    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        self.done = self.current_step >= self.n_windows
        self.truncated = self.balance <= 0.4 * self.initial_balance
        obs = self._next_observation()
        reward = self.balance - self.initial_balance
        return obs, reward, self.done, self.truncated, {}
    
    def _get_reward(self):
        current_window = self.data[self.current_step]
        close_idx = self.data_handler.feature_names.index('close')
        current_price = current_window[-1, close_idx]
        reward = 0
        if self.position == 1:
            reward = (current_price - self.last_trade_price) * self.leverage
        elif self.position == -1:
            reward = (self.last_trade_price - current_price) * self.leverage
        self.profit += reward
        if self.balance <= 0.1 * self.initial_balance:
            self.done = True
        return reward
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Profit: {self.profit}')
        print(f'Done: {self.done}')
        print('---------------------------------')

    def close(self):
        pass

    def seed(self, seed=None):
        pass