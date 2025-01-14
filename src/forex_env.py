import gymnasium as gym
import numpy as np
import pandas as pd
from gym import spaces
from src.data_handler import DataHandler

class ForexEnv(gym.Env):
    def __init__(self, instrument, start_date, end_date, granularity, initial_balance=1000, leverage=50, window_size=10):
        super(ForexEnv, self).__init__()
        self.data_handler = DataHandler()
        self.data = self.data_handler.get_data(instrument, start_date, end_date, granularity)
        self.data = self.data
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size ,len(self.data.columns)))
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.window_size = window_size
        self.position = 0
        self.last_trade_price = 0
        self.profit = 0
        self.done = False

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.profit = 0
        self.done = False
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        end = self.current_step + self.window_size
        obs = self.data.iloc[self.current_step:end].values
        return obs
    
    def _take_action(self, action):
        current_price = self.data['close'].values[self.current_step]
        if action == 0:
            self.position = 0
        elif action == 1:
            self.position = 1
            self.last_trade_price = current_price
        elif action == 2:
            self.position = -1
            self.last_trade_price = current_price
        else:
            raise ValueError(f"Invalid action: {action}")
        
    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.data) - self.window_size:
            self.done = True
        obs = self._next_observation()
        reward = self._get_reward()
        return obs, reward, self.done, {}
    
    def _get_reward(self):
        current_price = self.data['close'].values[self.current_step]
        reward = 0
        if self.position == 1:
            reward = (current_price - self.last_trade_price) * self.leverage
        elif self.position == -1:
            reward = (self.last_trade_price - current_price) * self.leverage
        self.profit += reward
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