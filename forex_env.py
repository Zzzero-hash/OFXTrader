import gym
from gym import spaces
import numpy as np
from functools import lru_cache
import pandas as pd  # Add this import
import logging

class OandaForexTradingEnv(gym.Env):
    """A custom environment for futures trading using technical indicators and market data."""
    def __init__(self, data_sequences, instrument="EUR_USD", initial_capital=1000, leverage=10, window_size=10, seed=None, render_interval=1000, stop_loss_percent=0.05, frequent_trade_penalty=0.5, risk_per_trade=0.02):
        """
        Initialize the environment.
        
        Parameters:
        - data_sequences: pandas DataFrame containing the market data and technical indicators.
        - instrument: The trading instrument (e.g., "EUR_USD").
        - initial_capital: The initial capital in units of currency.
        - leverage: The leverage factor to apply to trades.
        - window_size: The number of past observations to include in the state.
        - seed: Random seed for reproducibility.
        - render_interval: The interval at which to render the environment.
        - stop_loss_percent: The percentage threshold for stop loss.
        - frequent_trade_penalty: The penalty for frequent trades.
        - risk_per_trade: The percentage of capital to risk per trade.
        """
        super().__init__()
        
        # Market data
        self.data_sequences = data_sequences.values  # Convert DataFrame to numpy array once
        self.instrument = instrument
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage  # Initialize leverage
        self.window_size = window_size
        self.fee_rate = 0.0001
        self.percent_to_trade = 0.05  # Reduced position size
        self.position_size = 0
        self.trade_size = 0
        self.total_fees = 0
        self.total_profit = 0
        self.position = None
        self.entry_price = 0
        self.render_interval = render_interval
        self.stop_loss_percent = stop_loss_percent  # Set stop loss percentage
        self.frequent_trade_penalty = frequent_trade_penalty  # Initialize frequent trade penalty
        self.risk_per_trade = risk_per_trade  # Initialize risk per trade
        self.scaling_factor = 1e-4  # Add scaling factor for numerical stability
        self.max_portfolio_value = 1e7  # Maximum allowed portfolio value
        self.min_trade_size = 0.01  # Minimum trade size
        self.max_trade_size = initial_capital * 5  # Maximum trade size
        self.max_position_size = min(self.initial_capital * self.leverage * 0.95, 1e6)  # Calculate maximum position size
        self.position_size_limit = min(self.initial_capital * 10, 1e6)  # Set position size limit
        self.max_trades_per_day = 5  # Limit number of trades
        self.min_profit_threshold = 0.001  # Minimum profit to consider trade
        self.max_loss_threshold = -0.02  # Maximum loss before closing
        self.trades_today = 0
        self.last_trade_day = None
        
        # Risk management limits
        self.max_drawdown_limit = 0.15  # 15% maximum drawdown
        self.min_win_rate = 0.45  # Minimum required win rate
        
        # Trade tracking
        self.trade_log = []  # List to store individual trade information
        
        # Initialize trade_metrics as a dictionary
        self.trade_metrics = {
            'wins': 0,
            'losses': 0,
            'current_drawdown': 0,
            'max_drawdown': 0
        }
        
        # Action space definition: 0 = Hold, 1 = BUY, 2 = SELL, 3 = CLOSE_BUY, 4 = CLOSE_SELL
        self.action_space = spaces.Discrete(5)

        # Observation space definition: Using the shape of the DataFrame to set the limits
        self.observation_space = spaces.Box(low=-5, high=5, shape=(window_size, data_sequences.shape[1]), dtype=np.float32)

        self.log_interval = max(1, len(self.data_sequences) // 10)

        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Pre-allocate numpy arrays for better performance
        self.state_buffer = np.zeros((window_size, data_sequences.shape[1]), dtype=np.float32)
        self.data_array = data_sequences  # Use the numpy array directly

        self.current_step = 0
        self.peak_capital = initial_capital  # Initialize peak capital
        self.trade_log = []  # Initialize trade log
        # Ensure all attributes are initialized before calling reset
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def calculate_percentage_change_from_entry(self, entry_price, current_price):
        """Calculate percentage change with protection against division by zero."""
        if entry_price is None or entry_price == 0:
            return 0.0
        return (current_price - entry_price) / entry_price
        return (current_price - entry_price) / entry_price

    def apply_risk_management(self, current_price):
        """Simplified risk management rules."""
        if current_price <= 0:
            logging.warning("Current price is non-positive. Skipping risk management.")
            return 0.0
        
        # Simplified position size calculation
        risk_amount = self.capital * self.risk_per_trade
        position_size = (risk_amount / (current_price * self.stop_loss_percent)) * self.leverage
        
        return np.clip(position_size, self.min_trade_size, self.position_size_limit)

    def check_stop_loss(self, current_price):
        """Check if stop loss has been triggered and calculate profit/loss."""
        if not self.position or not self.entry_price:
            return False, 0.0

        price_change_percent = self.calculate_percentage_change_from_entry(self.entry_price, current_price)
        is_stopped = False
        profit = 0.0

        if self.position == 'Long':
            is_stopped = price_change_percent < -self.stop_loss_percent
            if is_stopped:
                profit = (current_price - self.entry_price) * self.position_size
                self.close_position()
        elif self.position == 'Short':
            is_stopped = price_change_percent > self.stop_loss_percent
            if is_stopped:
                profit = (self.entry_price - current_price) * self.position_size
                self.close_position()

        return is_stopped, profit

    def normalize_state(self, state):
        """Normalize state values to prevent numerical instability."""
        return np.clip(state / self.scaling_factor, -5, 5)

    def normalize_reward(self, reward):
        """Normalize reward to prevent extreme values."""
        return np.clip(reward, -1, 1)

    def step(self, action):
        if action not in range(self.action_space.n):
            raise ValueError(f"Invalid action: {action}")

        self.trade_size = 0
        previous_step = self.current_step if self.current_step != 0 else 0  # Track the previous step
        try:
            current_price = self.data_sequences[self.current_step, self.data_sequences.columns.get_loc('close')]
            previous_portfolio_value = min(self.capital + self.position_size * current_price, 1e9)
            reward = 0
            done = False
            truncated = False
            info = {}
            transaction_cost = 0
            slippage = 0

            if action == 0:  # Hold
                pass

            elif action == 1 and not self.position:  # BUY
                position_size = self.apply_risk_management(current_price)
                if position_size > 0:
                    self.position = 'Long'
                    self.entry_price = current_price
                    self.position_size = position_size
                    self.trade_size += 1
                    self.trade_log.append({'type': 'BUY', 'price': current_price, 'size': position_size})
                    logging.info(f"Executed BUY at {current_price} with size {position_size}")

            elif action == 2 and not self.position:  # SELL
                position_size = self.apply_risk_management(current_price)
                if position_size > 0:
                    self.position = 'Short'
                    self.entry_price = current_price
                    self.position_size = position_size
                    self.trade_size += 1
                    self.trade_log.append({'type': 'SELL', 'price': current_price, 'size': position_size})
                    logging.info(f"Executed SELL at {current_price} with size {position_size}")

            elif action == 3 and self.position == 'Long':  # CLOSE_BUY
                is_stopped, profit = self.check_stop_loss(current_price)
                if not is_stopped:
                    profit = (current_price - self.entry_price) * self.position_size
                    self.close_position()
                    logging.info(f"Closed BUY at {current_price} with profit {profit}")

            elif action == 4 and self.position == 'Short':  # CLOSE_SELL
                is_stopped, profit = self.check_stop_loss(current_price)
                if not is_stopped:
                    profit = (self.entry_price - current_price) * self.position_size
                    self.close_position()
                    logging.info(f"Closed SELL at {current_price} with profit {profit}")

            # Replace the existing stop-loss logic with the new method
            if self.position:
                is_stopped, stop_loss_profit = self.check_stop_loss(current_price)
                if is_stopped:
                    self.capital += stop_loss_profit + (self.position_size / self.leverage)
                    self.total_profit += stop_loss_profit
                    info = {
                        'type': f'Auto Close {self.position}',
                        'entry_price': float(self.entry_price),
                        'exit_price': float(current_price),
                        'profit': float(stop_loss_profit)
                    }
                    self.position = None
                    self.position_size = 0
                    self.entry_price = 0

            self.peak_capital = max(self.peak_capital, self.capital)
            minimum_capital_threshold = self.initial_capital * 0.5

            if self.capital < minimum_capital_threshold:
                done = True
                truncated = True
                logging.error("Truncated threshold reached!")
                self.render()
            self.prev_price = current_price

            unrealized_pnl = 0
            if self.position == 'Long':
                # Unrealized PnL for a long position
                unrealized_pnl = (current_price - self.entry_price) * self.position_size / self.leverage
            elif self.position == 'Short':
                # Unrealized PnL for a short position
                unrealized_pnl = (self.entry_price - current_price) * self.position_size / self.leverage

            # Add unrealized PnL to reward
            reward += unrealized_pnl * 0.0001  # Scale factor

            # Update trade metrics
            if info.get('profit', 0) > 0:
                self.trade_metrics['wins'] += 1
            elif info.get('profit', 0) < 0:
                self.trade_metrics['losses'] += 1

            # Append trade info to trade_log
            if info:
                self.trade_log.append(info)

            # Calculate drawdown
            self.trade_metrics['current_drawdown'] = (self.peak_capital - self.capital) / self.peak_capital
            self.trade_metrics['max_drawdown'] = max(self.trade_metrics['max_drawdown'],
                                                     self.trade_metrics['current_drawdown'])

            # Force close position if max loss exceeded
            if self.trade_metrics['current_drawdown'] > self.max_drawdown_limit:
                self.position = None
                self.position_size = 0
                done = True

        except IndexError as e:
            done = True
            truncated = False
            info = {'error': f"IndexError: {e} at step: {self.current_step}"}
            self.current_step = max(0, len(self.data_sequences) - 1)  # Ensure current_step is within bounds

        # Modify reward calculation
        portfolio_value = np.clip(
            self.capital + (self.position_size * current_price if self.position else 0),
            0,
            self.max_portfolio_value
        )

        # Calculate percentage change with scaling
        if previous_portfolio_value == 0:
            reward = 0
        else:
            reward = ((portfolio_value - previous_portfolio_value) /
                      max(previous_portfolio_value, self.initial_capital)) * self.scaling_factor

        # Normalize reward
        reward = self.normalize_reward(reward)

        # Add penalty for excessive position size
        if self.position_size > self.position_size_limit:
            reward -= 0.05  # Lower penalty for excessive position size

        # Ensure current_step does not exceed the length of data_sequences
        if self.current_step + 1 >= len(self.data_sequences):
            done = True
            truncated = False
            self.current_step = len(self.data_sequences) - 1  # Ensure current_step is within bounds
        else:
            self.current_step += 1

        # Normalize state
        if not done:
            self.state = self.normalize_state(self._next_observation())
        else:
            self.state = None

        if self.current_step % self.render_interval == 0 or done or truncated:
            logging.info(f"Rendering at step {self.current_step}")
            self.render()

        return self.state, reward, done, truncated, info

    def reset(self, *, seed=None):
        """
        Reset the environment to its initial state.

        Parameters:
        - seed: Random seed for reproducibility.

        Returns:
        - state: The initial state of the environment.
        - info: Additional information (empty dictionary).
        """
        if seed is not None:
            self.seed(seed)
        self.position_size = 0
        self.entry_price = 0
        self.trade_log = []
        self.total_fees = 0
        self.current_step = 0
        self.capital = self.initial_capital
        self.total_profit = 0
        self.current_step = self.window_size  # Start after the initial window
        self.state = self.normalize_state(self._next_observation())

        # Ensure current_step does not exceed the length of data_sequences
        if self.current_step >= len(self.data_sequences):
            self.current_step = len(self.data_sequences) - 1  # Ensure current_step is within bounds

        # Add a condition to prevent excessive logging
        if self.current_step % self.render_interval == 0:
            logging.info("Env Reset Called!")
        
        return self.state, {}

    def _next_observation(self):
        """Extracted logic from reset for clarity."""
        try:
            if self.current_step < self.window_size or self.current_step >= len(self.data_array):
                raise ValueError("Invalid current_step in _next_observation")
            self.state_buffer[:] = self.data_array[self.current_step - self.window_size : self.current_step]
            return self.state_buffer
        except Exception as e:
            logging.error(f"Error in _next_observation: Invalid current_step value: {self.current_step}, Error: {e}")
            raise e

    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}, Position: {self.position}, Capital: {self.capital:.2f}, Peak Capital: {self.peak_capital:.2f}")
        if self.trade_log:
            last_trade = self.trade_log[-1]
            exit_price = last_trade.get('exit_price', None)
            profit = last_trade.get('profit', None)
            
            exit_price_str = f"{exit_price:.2f}" if isinstance(exit_price, (int, float)) else "N/A"
            profit_str = f"{profit:.2f}" if isinstance(profit, (int, float)) else "N/A"
        
            entry_price = last_trade.get('entry_price', None)
            entry_price_str = f"{entry_price:.2f}" if isinstance(entry_price, (int, float)) else "N/A"
            print(f"Last Trade - Type: {last_trade['type']}, Entry: {entry_price_str}, Exit: {exit_price_str}, Profit: {profit_str}")
    
        closing_trades = [trade for trade in self.trade_log if trade.get('type') in ['CLOSE_BUY', 'CLOSE_SELL']]
        number_of_closing_trades = len(closing_trades)
    
        print(f"Total Profit: {self.total_profit:.2f}, Total Transaction Costs: {self.total_fees:.2f}, Number of Trades: {number_of_closing_trades}")
        
        winning_trades = [t for t in self.trade_log if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trade_log if t.get('profit', 0) < 0]
        trades_with_profit = [t for t in self.trade_log if t.get('profit') is not None]
        if trades_with_profit:
            average_profit = sum([t['profit'] for t in trades_with_profit]) / len(trades_with_profit)
            print(f"Total Winning Trades: {len(winning_trades)}, Total Losing Trades: {len(losing_trades)}")
            print(f"Average Profit per Trade: {average_profit:.2f}")

    def close(self):
        self.data_sequences = None
        self.instrument = None
        self.initial_capital = 0
        self.capital = 0
        self.leverage = 0
        self.window_size = 0
        self.fee_rate = 0
        self.percent_to_trade = 0
        self.position_size = 0
        self.trade_size = 0
        self.total_fees = 0
        self.total_profit = 0
        self.position = None
        self.entry_price = 0
        self.render_interval = 0
        self.stop_loss_percent = 0
        self.frequent_trade_penalty = 0
        self.risk_per_trade = 0
        self.action_space = None
        self.observation_space = None
        self.log_interval = 0
        self.np_random = None
        self.state_buffer = None
        self.data_array = None
        self.current_step = 0
        self.trade_log = None
        self.peak_capital = 0
        self.state = None
        self.prev_price = 0

    def close_position(self):
        """Close the current position and reset related variables."""
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.trade_size = 0
        self.trades_today = 0
        self.last_trade_day = None