import gym
from gym import spaces
import numpy as np

class ForexTradingEnv(gym.Env):
    """A custom environment for futures trading using technical indicators and market data."""
    def __init__(self, data_sequences, initial_capital=1000, leverage=50, window_size=10, seed=None, render_interval=1000, stop_loss_percent=0.02):
        """
        Initialize the environment.
        
        Parameters:
        - data_sequences: DataFrame containing the market data and technical indicators.
        - initial_capital: The initial capital in units of currency.
        - leverage: The leverage factor to apply to trades.
        - window_size: The number of past observations to include in the state.
        - seed: Random seed for reproducibility.
        - render_interval: The interval at which to render the environment.
        - stop_loss_percent: The percentage threshold for stop loss.
        """
        super(ForexTradingEnv, self).__init__()
        
        # Market data
        self.data_sequences = data_sequences
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.window_size = window_size
        self.fee_rate = 0.0001
        self.percent_to_trade = 0.1
        self.position_size = 0
        self.trade_size = 0
        self.total_fees = 0
        self.total_profit = 0
        self.position = None
        self.entry_price = 0
        self.render_interval = render_interval
        self.stop_loss_percent = stop_loss_percent
        
        # Action space definition: 0 = Hold, 1 = Open Long, 2 = Open Short, 3 = Close Long, 4 = Close Short
        self.action_space = spaces.Discrete(5)

        # Observation space definition: Using the shape of the DataFrame to set the limits
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, data_sequences.shape[1]), dtype=np.float32)

        self.log_interval = max(1, len(self.data_sequences) // 10)

        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.reset(seed=seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def calculate_percentage_change_from_entry(self, entry_price, current_price):
        absolute_difference = abs(entry_price - current_price)
        percentage_change = (absolute_difference / entry_price) if entry_price != 0 else 0
        return percentage_change
        
    def step(self, action):
        if action not in range(self.action_space.n):
            raise ValueError(f"Invalid action: {action}. It should be within the range of 0 to {self.action_space.n - 1}.")
        try:
            if self.current_step < len(self.data_sequences):
                current_price = self.data_sequences.iloc[self.current_step]['close']
            else:
                done = True
                truncated = False
                print("Done reached!")
                return self.state, 0, done, truncated, {}

            if self.current_step >= len(self.data_sequences) - 1:
                done = True
                truncated = False
                print("Done reached!")
            previous_portfolio_value = self.capital + self.position_size * current_price
            reward = 0
            done = False
            truncated = False
            info = {}

            transaction_cost = 0
            slippage = 0
    
            if action == 0:  # Hold
                pass
    
            elif action == 1 and not self.position:  # Open Long
                self.trade_size = self.capital * self.percent_to_trade
                transaction_cost = self.trade_size * self.fee_rate
                slippage = self.trade_size * 0.0002
                required_margin = (self.trade_size * self.leverage) / self.leverage
                # Check if we have enough capital to open the trade
                if self.capital < (transaction_cost + required_margin + slippage):
                    reward -= 2.0  # Punish invalid trade attempt
                    info = {'type': 'Invalid Trade', 'reason': 'Insufficient Margin'}
                else:
                    self.position = 'Long'
                    self.trade_size = self.capital * self.percent_to_trade
                    transaction_cost = self.trade_size * self.fee_rate
                    slippage = self.trade_size * 0.0002
                    self.position_size = self.trade_size * self.leverage
                    required_margin = self.position_size / self.leverage
                    self.entry_price = current_price
                    self.capital -= transaction_cost + required_margin
                    self.total_fees += transaction_cost
                    info = {'type': 'Open Long',
                                'entry_price': float(current_price),
                                'position_size': float(self.position_size),
                                'capital': float(self.capital),
                                'transaction_cost': float(transaction_cost)
                            }
    
            elif action == 2 and not self.position:  # Open Short
                self.trade_size = self.capital * self.percent_to_trade
                transaction_cost = self.trade_size * self.fee_rate
                slippage = self.trade_size * 0.0002
                required_margin = (self.trade_size * self.leverage) / self.leverage
                # Check if we have enough capital to open the trade
                if self.capital < (transaction_cost + required_margin + slippage):
                    reward -= 2.0  # Punish invalid trade attempt
                    info = {'type': 'Invalid Trade', 'reason': 'Insufficient Margin'}
                else:
                    self.position = 'Short'
                    self.trade_size = self.capital * self.percent_to_trade
                    transaction_cost = self.trade_size * self.fee_rate
                    slippage = self.trade_size * 0.0002
                    self.position_size = self.trade_size * self.leverage
                    required_margin = self.position_size / self.leverage
                    self.entry_price = current_price
                    self.capital -= transaction_cost + required_margin
                    self.total_fees += transaction_cost
                    info = {'type': 'Open Short', 
                                'entry_price': float(current_price),
                                'position_size': float(self.position_size),
                                'capital': float(self.capital),
                                'transaction_cost': float(transaction_cost)
                            }
    
            elif action == 3 and self.position == 'Long':  # Close Long
                if self.position_size > 0:
                    percentage_change = self.calculate_percentage_change_from_entry(self.entry_price, current_price)
                    profit = percentage_change * self.position_size if current_price > self.entry_price else -percentage_change * self.position_size
                    self.capital += profit + (self.position_size / self.leverage)
                    self.total_profit += profit
                    reward = profit
                    info = {'type': 'Close Long', 
                            'entry_price': float(self.entry_price), 
                            'exit_price': float(current_price), 
                            'profit': float(profit)}
                    self.position = None
                    self.position_size = 0
                    self.entry_price = 0
                else:
                    reward = -1
    
            elif action == 4 and self.position == 'Short':  # Close Short
                if self.position_size > 0:
                    percentage_change = self.calculate_percentage_change_from_entry(self.entry_price, current_price)
                    profit = -percentage_change * self.position_size if current_price > self.entry_price else percentage_change * self.position_size
                    self.capital += profit + (self.position_size / self.leverage)
                    self.total_profit += profit
                    reward = profit
                    info = {'type': 'Close Short', 
                            'entry_price': float(self.entry_price), 
                            'exit_price': float(current_price), 
                            'profit': float(profit)}
                    self.position = None
                    self.position_size = 0
                    self.entry_price = 0
                else:
                    reward = -1

            if self.position == 'Long':
                if (self.entry_price - current_price) / self.entry_price > self.stop_loss_percent:
                    # Auto-close logic
                    percentage_change = self.calculate_percentage_change_from_entry(self.entry_price, current_price)
                    profit = percentage_change * self.position_size if current_price > self.entry_price else -percentage_change * self.position_size
                    self.capital += profit + (self.position_size / self.leverage)
                    self.total_profit += profit
                    reward = profit
                    info = {'type': 'Auto Close Long', 
                            'entry_price': float(self.entry_price), 
                            'exit_price': float(current_price), 
                            'profit': float(profit)}
                    self.position = None
                    self.position_size = 0
                    self.entry_price = 0
            elif self.position == 'Short':
                if (current_price - self.entry_price) / self.entry_price > self.stop_loss_percent:
                    # Auto-close logic
                    percentage_change = self.calculate_percentage_change_from_entry(self.entry_price, current_price)
                    profit = -percentage_change * self.position_size if current_price > self.entry_price else percentage_change * self.position_size
                    self.capital += profit + (self.position_size / self.leverage)
                    self.total_profit += profit
                    reward = profit
                    info = {'type': 'Auto Close Short', 
                            'entry_price': float(self.entry_price), 
                            'exit_price': float(current_price), 
                            'profit': float(profit)}
                    self.position = None
                    self.position_size = 0
                    self.entry_price = 0

            if action in [1, 2, 3, 4] and info:
                self.trade_log.append(info)
                
            self.peak_capital = max(self.peak_capital, self.capital)
            minimum_capital_threshold = self.initial_capital * 0.5
    
            if self.capital < minimum_capital_threshold:
                done = True
                truncated = True
                print("Truncated threshold reached!")
                self.render()
            
            self.prev_price = current_price
    
            self.current_step += 1

            unrealized_pnl = 0
            if self.position == 'Long':
                # Unrealized PnL for a long position
                unrealized_pnl = (current_price - self.entry_price) * self.position_size / self.leverage
            elif self.position == 'Short':
                # Unrealized PnL for a short position
                unrealized_pnl = (self.entry_price - current_price) * self.position_size / self.leverage

            # Add unrealized PnL to reward
            reward += unrealized_pnl * 0.0001  # Scale factor
    
            if self.current_step >= len(self.data_sequences):
                done = True
                print("Done reached!")
                
        except IndexError as e:
            print(f"IndexError: {e} at step: {self.current_step}")
            done = True
            truncated = False

        self.state = self._next_observation() if not done else None

        if self.current_step % self.render_interval == 0 or done or truncated:
            self.render()

        return self.state, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.position_size = 0
        self.entry_price = 0
        self.trade_log = []
        self.total_fees = 0
        self.peak_capital = self.initial_capital
        self.capital = self.initial_capital
        self.current_step = self.window_size  # Start after the initial window
        self.total_profit = 0
        self.position = None
        self.current_step = self.window_size  # Start after the initial window
        self.state = self._next_observation()

        print(f"Env Reset Called!")
        
        return self.state, {}
    
    def _next_observation(self):
        try:
            obs = self.data_sequences.iloc[self.current_step - self.window_size:self.current_step].values
            return np.array(obs, dtype=np.float32)
        except Exception as e:
            print(f"Error in _next_observation: {e}")
            print(f"Data sequences shape: {self.data_sequences.shape}, Observation length: {len(obs) if 'obs' in locals() else 'N/A'}")
            raise e
    
    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}, Position: {self.position}, Capital: {self.capital:.2f}, Peak Capital: {self.peak_capital:.2f}")
        if self.trade_log:
            last_trade = self.trade_log[-1]
            exit_price = last_trade.get('exit_price', None)
            profit = last_trade.get('profit', None)
            
            exit_price_str = f"{exit_price:.2f}" if isinstance(exit_price, (int, float)) else "N/A"
            profit_str = f"{profit:.2f}" if isinstance(profit, (int, float)) else "N/A"
        
            print(f"Last Trade - Type: {last_trade['type']}, Entry: {last_trade['entry_price']:.2f}, Exit: {exit_price_str}, Profit: {profit_str}")
    
        closing_trades = [trade for trade in self.trade_log if trade.get('type') in ['Close Long', 'Close Short']]
        number_of_closing_trades = len(closing_trades)
    
        print(f"Total Profit: {self.total_profit:.2f}, Total Transaction Costs: {self.total_fees:.2f}, Number of Trades: {number_of_closing_trades}")
        
        winning_trades = [t for t in self.trade_log if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trade_log if t.get('profit', 0) < 0]
        trades_with_profit = [t for t in self.trade_log if t.get('profit') is not None]
        if trades_with_profit:
            average_profit = sum(t.get('profit', 0) for t in trades_with_profit) / len(trades_with_profit)
        else:
            average_profit = 0.0
    
        print(f"Total Winning Trades: {len(winning_trades)}, Total Losing Trades: {len(losing_trades)}")
        print(f"Average Profit per Trade: {average_profit:.2f}")

    def close(self):
        self.data_sequences = None
        self.trade_log = None
        self.state = None
        pass