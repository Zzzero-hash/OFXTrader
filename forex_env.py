import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Tuple, Discrete
from collections import deque
from gymnasium.utils import seeding
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ForexEnv(gym.Env):
    def __init__(self, instruments, data_array, feature_names, initial_balance=1000, leverage=10, spread_pips=0.0001, max_steps=50000, render_frequency=1000):
        super(ForexEnv, self).__init__()
        self.instruments = instruments
        self.n_instruments = len(instruments)
        self.data_array = data_array
        self.feature_names = feature_names
        self.n_timesteps, self.n_features = self.data_array.shape

        # Debug output to understand data structure
        print(f"Feature names: {feature_names}")
        print(f"Data array shape: {data_array.shape}")
        print(f"Instruments: {instruments}")
        
        # Try to find close prices more robustly
        self.close_indices = {}
        
        # Method 1: Look for exact format '{instrument}_close'
        for instr in instruments:
            col_name = f"{instr}_close"
            if col_name in feature_names:
                self.close_indices[instr] = feature_names.index(col_name)
        
        # If not all instruments found, try method 2: Look for any columns containing 'close'
        if len(self.close_indices) < len(instruments):
            close_columns = [i for i, name in enumerate(feature_names) if 'close' in name.lower()]
            if close_columns:
                # Assign remaining instruments to available close columns
                remaining = [instr for instr in instruments if instr not in self.close_indices]
                for i, instr in enumerate(remaining):
                    self.close_indices[instr] = close_columns[i % len(close_columns)]
        
        # Method 3: If no 'close' found at all, assume last column for each instrument section
        if len(self.close_indices) < len(instruments):
            cols_per_instrument = len(feature_names) // len(instruments)
            for i, instr in enumerate(instruments):
                if instr not in self.close_indices:
                    # Assuming the last column in each instrument's features is the close price
                    self.close_indices[instr] = (i * cols_per_instrument) + (cols_per_instrument - 1)
        
        print(f"Found close indices: {self.close_indices}")
            
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.spread = spread_pips
        self.max_steps = max_steps
        self.render_frequency = render_frequency

        # Observation space: market data + positions + unrealized P/L for each instrument
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features + 2 * self.n_instruments,), dtype=np.float32)

        # Action space: Tuple of Discrete spaces for each instrument (0: close, 1: long, 2: short)
        self.action_space = Tuple([Discrete(3) for _ in range(self.n_instruments)])

    def reset(self, *, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.positions = [0] * self.n_instruments  # 0: no position, 1: long, -1: short
        self.entry_prices = [None] * self.n_instruments
        self.unrealized_pnls = [0.0] * self.n_instruments
        self.current_step = 0
        self.total_value = self.initial_balance
        self.past_changes = deque(maxlen=20)
        self.done = False
        self.truncated = False
        return self._get_observation(), {}

    def _get_observation(self):
        """Get the current observation including market data and portfolio state."""
        if self.current_step >= self.n_timesteps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        data = self.data_array[self.current_step]
        positions = np.array(self.positions, dtype=np.float32)
        unrealized_pnls = np.array(self.unrealized_pnls, dtype=np.float32)
        return np.concatenate([data, positions, unrealized_pnls])

    def step(self, action):
        # Convert tuple action to NumPy array
        action = np.array(action, dtype=np.int32)
        
        # Log the action for debugging
        logger.info(f"Action received: {action}")
        
        # Ensure action is within valid range [0, 2]
        if not all(0 <= a <= 2 for a in action):
            logger.warning(f"Clipping invalid action: {action}")
            action = np.clip(action, 0, 2)

        # Calculate the previous total before taking actions
        previous_total_value = self.total_value

        # Get current prices for each instrument
        current_prices = [self.data_array[self.current_step, self.close_indices[instr]] for instr in self.instruments]

        # Update unrealized P/L for open positions
        for i in range(self.n_instruments):
            if self.positions[i] == 1:
                self.unrealized_pnls[i] = (current_prices[i] - self.entry_prices[i]) * self.leverage
            elif self.positions[i] == -1:
                self.unrealized_pnls[i] = (self.entry_prices[i] - current_prices[i]) * self.leverage
            else:
                self.unrealized_pnls[i] = 0.0

        # Process actions for each instrument
        for i in range(self.n_instruments):
            if action[i] == 0:  # Close Position
                if self.positions[i] != 0:
                    profit = self.unrealized_pnls[i]
                    self.balance += profit
                    self.positions[i] = 0
                    self.entry_prices[i] = None
                    self.unrealized_pnls[i] = 0.0
            elif action[i] == 1:  # Go Long
                if self.positions[i] == -1:
                    profit = self.unrealized_pnls[i]
                    self.balance += profit
                    self.positions[i] = 0
                    self.entry_prices[i] = None
                if self.positions[i] != 1:
                    self.entry_prices[i] = current_prices[i] + self.spread
                    self.positions[i] = 1
            elif action[i] == 2:  # Go Short
                if self.positions[i] == 1:
                    profit = self.unrealized_pnls[i]
                    self.balance += profit
                    self.positions[i] = 0
                    self.entry_prices[i] = None
                if self.positions[i] != -1:
                    self.entry_prices[i] = current_prices[i] - self.spread
                    self.positions[i] = -1

        # Recalculate unrealized P/L after actions
        for i in range(self.n_instruments):
            if self.positions[i] == 1:
                self.unrealized_pnls[i] = (current_prices[i] - self.entry_prices[i]) * self.leverage
            elif self.positions[i] == -1:
                self.unrealized_pnls[i] = (self.entry_prices[i] - current_prices[i]) * self.leverage
            else:
                self.unrealized_pnls[i] = 0.0

        # Compute total portfolio value and reward
        self.total_value = self.balance + sum(self.unrealized_pnls)
        change = self.total_value - previous_total_value
        self.past_changes.append(change)
        std_dev = np.std(self.past_changes) if len(self.past_changes) > 1 else 0.0
        k = 0.1  # Risk aversion parameter
        reward = change - k * std_dev

        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        self.truncated = self.balance <= 0.5 * self.initial_balance
        if self.render_frequency > 0 and self.current_step % self.render_frequency == 0:
            self.render()
        observation = self._get_observation()
        info = {'balance': self.balance, 'total_value': self.total_value, 'positions': self.positions}
        return observation, reward, self.done, self.truncated, info

    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Total Value: {self.total_value:.2f}')
        print(f'Positions: {self.positions}')
        print('---------------------------------')

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]