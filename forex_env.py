import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data_handler import DataHandler
from gymnasium.envs.registration import register

class ForexEnv(gym.Env):
    def __init__(self, instrument="EUR_USD", start_date="2022-01-01", end_date="2023-01-01", 
                 granularity="M1", initial_balance=1000, leverage=50, window_size=14, spread_pips=0.0001):
        super(ForexEnv, self).__init__()
        self.data_handler = DataHandler()
        self.data, self.feature_names = self.data_handler.get_data(instrument, start_date, 
                                                                  end_date, granularity, window_size)
        self.n_windows, self.window_size, self.n_features = self.data.shape
        self.close_idx = self.feature_names.index('close')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(window_size, self.n_features))
        self.action_space = spaces.Discrete(3)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.window_size = window_size
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = None
        self.entry_step = 0
        self.spread = spread_pips
        self.current_step = 0
        self.last_profit = 0
        self.done = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.entry_step = 0
        self.last_profit = 0
        self.current_step = 0
        self.done = False
        self.truncated = False
        return self._next_observation(), {}

    def _next_observation(self):
        if self.current_step >= self.n_windows:
            self.done = True
            return np.zeros((self.window_size, self.n_features))
        return self.data[self.current_step]

    def _calculate_profit(self, exit_price):
        if self.position == 0:
            return 0  # No position to close
        
        if self.position == 1:
            profit = (exit_price - self.entry_price) * self.leverage
        else:
            profit = (self.entry_price - exit_price) * self.leverage
        
        self.balance += profit
        self.position = 0
        self.entry_price = None
        return profit

    def _take_action(self, action):
        current_price = self.data[self.current_step, -1, self.close_idx]
        self.last_profit = 0

        # Close position or switch
        if action == 0:
            if self.position != 0:
                self.last_profit = self._calculate_profit(current_price)
        else:
            # Handle position switching
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
        prev_balance = self.balance
        self._take_action(action)
        self.current_step += 1

        # Calculate components
        time_penalty = 0.0
        if self.position != 0:
            time_penalty = 0.02 * (self.current_step - self.entry_step)  # 0.02 per step penalty
            
        drawdown_penalty = 0.0
        if self.balance <= 0.4 * self.initial_balance:
            drawdown_penalty = -50.0  # Large penalty for account drawdown
            self.truncated = True

        # Calculate raw reward components
        reward = (
            self.last_profit 
            - time_penalty 
            + drawdown_penalty
        )

        # Normalize reward to percentage of initial balance
        reward_normalized = reward / self.initial_balance

        # Update termination conditions
        self.done = self.current_step >= self.n_windows
        self.truncated = self.truncated or (self.balance <= 0.4 * self.initial_balance)

        return (
            self._next_observation(),
            reward_normalized,
            self.done,
            self.truncated,
            {'balance': self.balance, 'position': self.position}
        )

    # Remaining methods (render, close, seed) remain unchanged
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Position: {self.position}')
        print(f'Last Profit: {self.last_profit:.2f}')
        print('---------------------------------')

register(
    id='forex-v0',
    entry_point='forex_env:ForexEnv'
)