import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data_handler import DataHandler
from gymnasium.envs.registration import register
import gymnasium.utils.seeding as seeding  # For reproducibility

class ForexEnv(gym.Env):
    """
    A Forex trading environment for reinforcement learning.
    
    Expects data as a raw 2D array of shape (n_timesteps, n_features) after technical
    analysis and cleanup. Each row corresponds to one time step's observation.
    RLlib’s built-in recurrent wrappers will batch individual time steps into sequences.
    
    The action space is Discrete(3):
        0: Close any open position.
        1: Enter (or switch to) a long position.
        2: Enter (or switch to) a short position.
    """
    def __init__(self, instrument="EUR_USD", start_date="2022-01-01", end_date="2023-01-01", 
                 granularity="M1", initial_balance=1000, leverage=50, window_size=14, 
                 spread_pips=0.0001, render_frequency=1000):
        super(ForexEnv, self).__init__()
        
        # Fetch and process data using DataHandler.
        self.data_handler = DataHandler()
        self.data, self.feature_names = self.data_handler.get_data(instrument, start_date, 
                                                                  end_date, granularity, window_size)
        if self.data.size == 0:
            raise ValueError("No data fetched. Check your DataHandler configuration.")
        
        # Expect data shape: (n_timesteps, n_features)
        try:
            self.n_timesteps, self.n_features = self.data.shape
        except ValueError:
            raise ValueError(f"Unexpected data shape: {self.data.shape}. Expected (n_timesteps, n_features).")
        
        # Find the index of the 'close' price (for profit calculation)
        try:
            self.close_idx = self.feature_names.index('close')
        except ValueError:
            raise ValueError("'close' not found in feature names.")
        
        # Define observation and action spaces.
        # Now each observation is a 1D vector (n_features,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.n_features,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: close, 1: long, 2: short
        
        # Ensure data is in float32.
        self.data = self.data.astype(np.float32)
        
        # Trading parameters.
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.spread = spread_pips
        self.render_frequency = render_frequency  # Controls rendering frequency.
        
        # Position management.
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = None
        self.entry_step = 0
        
        # Tracking.
        self.current_step = 0
        self.total_profit = 0
        self.done = False
        self.truncated = False

    def reset(self, *, seed=None, options=None):
        self.seed(seed)
        # Reset trading state.
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
        """
        Returns the next observation (a single time step).
        If the current step exceeds the number of timesteps, returns a zero vector.
        """
        if self.current_step >= self.n_timesteps:
            self.done = True
            return np.zeros((self.n_features,), dtype=np.float32)
        return self.data[self.current_step].astype(np.float32)

    def _calculate_profit(self, exit_price):
        """
        Calculates profit based on the current position and exit price.
        Updates balance and resets the position.
        """
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
        """
        Executes the action:
            0: Close position.
            1: Enter long (or switch from short).
            2: Enter short (or switch from long).
        """
        # Use the current time step's close price.
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
        """
        Executes one time step.
        Returns (observation, reward, done, truncated, info).
        """
        self._take_action(action)
        self.current_step += 1

        # Time penalty for holding a position.
        time_penalty = 0.0
        if self.position != 0:
            time_penalty = 0.02 * (self.current_step - self.entry_step)
        
        # Drawdown penalty.
        drawdown_penalty = 0.0
        if self.balance <= 0.4 * self.initial_balance:
            drawdown_penalty = -50.0
            self.truncated = True

        reward = self.last_profit - time_penalty + drawdown_penalty
        reward_normalized = reward / self.initial_balance

        if self.current_step >= self.n_timesteps:
            self.done = True
        if self.truncated or (self.balance <= 0.4 * self.initial_balance):
            self.truncated = True

        if self.render_frequency > 0 and self.current_step % self.render_frequency == 0:
            self.render()

        return self._next_observation(), reward_normalized, self.done, self.truncated, {'balance': self.balance, 'position': self.position}

    def render(self, mode='human'):
        """
        Prints a summary of the current state.
        """
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Position: {self.position}')
        print(f'Total Profit + (Unrealized P/L): {self.total_profit:.2f}')
        print('---------------------------------')

    def close(self):
        """
        Clean up resources if needed.
        """
        pass

    def seed(self, seed=None):
        """
        Sets the random seed for reproducibility.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Register the environment.
register(
    id='forex-v0',
    entry_point='forex_env:ForexEnv'
)
