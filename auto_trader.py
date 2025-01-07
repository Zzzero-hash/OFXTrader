import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import datetime
from stable_baselines3 import PPO
import forex_env
import ta  # Technical Analysis library
from finrl.config_tickers import FX_TICKER
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS

def fetch_oanda_candles_chunk(instrument, from_time, to_time, granularity, access_token):
    """Fetch up to `count` candles in one request."""
    client = oandapyV20.API(access_token=access_token)
    params = {
        "from": pd.Timestamp(from_time).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "to": pd.Timestamp(to_time).strftime('%Y-%m-%dT%H:%M:%SZ'),      
        "granularity": granularity
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    resp = client.request(r)
    
    # Parse into DataFrame
    data = resp["candles"]
    df = pd.DataFrame([{
        'time': candle['time'],
        'open': float(candle['mid']['o']),
        'high': float(candle['mid']['h']),
        'low': float(candle['mid']['l']),
        'close': float(candle['mid']['c']),
        'volume': int(candle['volume'])
    } for candle in data])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    return df

def fetch_oanda_candles_range(instrument, start_date, end_date, granularity, access_token):
    current_start = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    granularity_map = {
        "M1": pd.Timedelta(minutes=1),
        "M5": pd.Timedelta(minutes=5),
        "M15": pd.Timedelta(minutes=15),
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
        "D": pd.Timedelta(days=1)
    }
    granularity_freq = granularity_map.get(granularity, pd.Timedelta(minutes=1))
    time_step = 5000 * granularity_freq

    df = pd.DataFrame()

    while current_start < end_date:
        to_dt = min(current_start + time_step, end_date)  # Don't exceed end_date
        df_chunk = fetch_oanda_candles_chunk(
            instrument=instrument,
            from_time=current_start,
            to_time=to_dt,
            granularity=granularity,
            access_token=access_token,
        )
        df = pd.concat([df, df_chunk], ignore_index=True)
        current_start = to_dt  # Move the current start forward
    return df

def add_technical_indicators(df):
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['dx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    bb = ta.volatility.BollingerBands(df['close'], window=14)
    df['bb_ub'] = bb.bollinger_hband()
    df['bb_lb'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def train_model(train_data, timesteps=50000, stop_loss_percent=0.02):
    env_train = forex_env.ForexTradingEnv(
        data_sequences=train_data,
        stop_loss_percent=stop_loss_percent
    )
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo")
    trained_model = agent.train_model(model=model, tb_log_name='ppo', total_timesteps=timesteps)
    return trained_model

def test_model(test_data, model):
    env_test = forex_env.ForexTradingEnv(data_sequences=test_data)
    obs, info = env_test.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(np.array(obs))
        obs, reward, done, truncated, info = env_test.step(action)
        total_reward += reward
    return total_reward

def check_performance(env, threshold=0.9):
    """
    Evaluate model performance. If performance is below threshold, return False.
    Replace this logic with your own performance metrics.
    """
    performance_score = 1.0  # Stub for actual evaluation
    return performance_score >= threshold

def auto_retrain(instrument, granularity, start_date, end_date, access_token, timesteps=50000):
    """
    Retrain model by fetching new data and using FinRL's DRLAgent.
    """
    new_data = fetch_oanda_candles_range(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        access_token=access_token
    )
    new_data = add_technical_indicators(new_data)
    train_data = new_data[:int(0.8 * len(new_data))]
    new_model = train_model(train_data, timesteps=timesteps)
    return new_model

def run_bot_with_monitoring(instrument, granularity, start_date, end_date, access_token):
    """
    Main loop that executes trades and checks performance regularly.
    """
    df = fetch_oanda_candles_range(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        access_token=access_token,
        granularity=granularity
    )
    df = add_technical_indicators(df)
    train_data = df[:int(0.8 * len(df))]
    test_data = df[int(0.8 * len(df)):]

    model = train_model(train_data)
    while True:
        test_reward = test_model(test_data, model)
        if not check_performance(env=None):  # Pass appropriate env or data
            print("Performance below threshold, retraining...")
            model = auto_retrain(instrument, granularity, start_date, end_date, access_token)
        # Insert actual trade execution calls here

# Example usage
instrument = 'EUR_USD'
granularity = 'M5'
start_date = '2019-01-01T00:00:00Z'
end_date = '2024-12-31T00:00:00Z'
access_token = 'a15df916d468a21855b25932c59b6947-de38c8e63794cf1d040c170d1ca6df24'

run_bot_with_monitoring(instrument, granularity, start_date, end_date, access_token)