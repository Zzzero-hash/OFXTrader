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
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data = self.data.set_index('time')
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size ,len(self.data.columns)))
        self.current_step = 0