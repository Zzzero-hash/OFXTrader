# ...existing code...

import gym
import numpy as np

class DIEngineForexEnv(gym.Env):
    def __init__(self, df, window_size=10, frame_bound=(10, None), unit_side='right'):
        super(DIEngineForexEnv, self).__init__()
        # Store input data
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.unit_side = unit_side
        self.initial_capital = 1000.0
        self.current_capital = self.initial_capital
        self.transaction_cost = 0.001
        self.open_position = None
        self.open_price = 0.0
        self.action_space = gym.spaces.Discrete(3)  # Example: [Hold, Buy, Sell]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(df)), dtype=np.float32
        )
        self.position_history = []
        self.profit_history = []
        self.raw_prices = df['Close'].values
        self._env_id = 0
        self.save_path = "plots/"

    def reset(self):
        self.current_step = self.frame_bound[0]
        self.current_capital = self.initial_capital
        self.open_position = None
        self.open_price = 0.0
        self.position_history = []
        self.profit_history = []
        self._start_tick = self.frame_bound[0]
        self._end_tick = self.frame_bound[1] if self.frame_bound[1] else len(self.df) - 1
        # Return initial observation
        return self._get_observation()

    def step(self, action):
        self.position_history.append(action)
        old_capital = self.current_capital
        self._update_position(action)
        reward = self._calculate_reward(old_capital)
        self.current_step += 1
        done = self.current_step >= (self.frame_bound[1] if self.frame_bound[1] else len(self.df))
        self.profit_history.append(self.current_capital)
        return self._get_observation(), reward, done, {}

    def _update_position(self, action):
        # Example trade logic
        current_price = self.df['Close'].iloc[self.current_step]
        if action == 1 and self.open_position is None:
            self.open_position = 'long'
            self.open_price = current_price
        elif action == 2 and self.open_position is None:
            self.open_position = 'short'
            self.open_price = current_price
        elif action == 0 and self.open_position is not None:
            price_diff = (current_price - self.open_price) if self.open_position == 'long' else (self.open_price - current_price)
            self.current_capital += price_diff - abs(price_diff) * self.transaction_cost
            self.open_position = None

    def _calculate_reward(self, old_capital):
        return self.current_capital - old_capital

    def render(self) -> None:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self.profit_history)
        plt.savefig(self.save_path + str(self._env_id) + "-profit.png")

        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(len(self.position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        plt.plot(eps_price)

        short_ticks = []
        long_ticks = []
        hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if self.position_history[i] == 2:  # short
                short_ticks.append(tick)
            elif self.position_history[i] == 1:  # long
                long_ticks.append(tick)
            else:
                hold_ticks.append(tick)  # hold

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=12, color='green', label='Long')
        plt.plot(hold_ticks, eps_price[hold_ticks], 'b^', markersize=12, color='blue', label='Hold')
        plt.plot(short_ticks, eps_price[short_ticks], 'r^', markersize=12, color='red', label='Short')
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.save_path + str(self._env_id) + "-price.png")

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size : self.current_step].values
    
    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

def create_di_env(df, window_size=10, frame_bound=(10, None)):
    """Create a DI-Engine compatible forex environment."""
    return DIEngineForexEnv(
        df=df,
        window_size=window_size,
        frame_bound=frame_bound,
        unit_side='right'
    )
