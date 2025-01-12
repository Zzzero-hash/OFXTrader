from gym_anytrading.envs import ForexEnv
import gymnasium as gym
import logging

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]  # Ensure capitalized
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]  # Ensure capitalized
    return prices, signal_features

class MyForexEnv(ForexEnv):
    _process_data = my_process_data

    def step(self, action):
        logging.debug(f"[Env] Taking action: {action}")
        # Note: super() may return (obs, reward, terminated, truncated, info)
        observation, reward, terminated, truncated, info = super().step(action)
        logging.debug(f"[Env] Obs: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, info

def create_env(df, window_size=10, frame_bound=None, unit_side='right'):
    """
    Create a MyForexEnv environment with the given DataFrame and parameters.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    window_size (int): The window size for the environment.
    frame_bound (tuple): The frame boundaries for the environment.
    unit_side (str): The unit side for the environment.

    Returns:
    env: The created environment.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")
    
    if frame_bound is None:
        frame_bound = (window_size, len(df))

    env = gym.make('forex-v0',
                   df=df,
                   window_size=window_size,
                   frame_bound=frame_bound,
                   unit_side=unit_side)
    return env
