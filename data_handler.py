import oandapyV20
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from oandapyV20.endpoints.instruments import InstrumentsCandles
import logging
import time
from ta import add_all_ta_features
from key_handler import decrypt_data  # Updated import statement

def fetch_chunk(params, client):
    for _ in range(3):  # Retry fetching data up to 3 times
        try:
            request_params = params.copy()
            instrument = request_params.pop('instrument')
            logging.info(f"Fetching data with params: {params}")
            response = client.request(InstrumentsCandles(instrument=instrument, params=request_params))
            logging.info(f"Response received: {response}")
            candles = response.get('candles', [])
            if not candles:
                logging.warning(f'EOD reached for {params["from"]}')
                return pd.DataFrame()
            df = pd.DataFrame([{
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': candle['volume']
            } for candle in candles])
            df['time'] = pd.to_datetime(df['time']).dt.tz_convert('UTC')
            return df
        except KeyError as e:
            logging.error(f"Missing key: {e}. Params: {params}")
            return pd.DataFrame()
        except Exception as e:
            logging.warning(f"Retry {_+1}/3 failed: {e}")
            time.sleep(2)
    return pd.DataFrame()

class DataHandler:
    def __init__(self):
        # api_key, account = decrypt_data()
        self.api_key = '7d72ad59524a9f896c85eb7cc9d21a37-aa2a02a5157b1891340ac08e9a1d1c29'
        self.account = '101-001-23675199-001'
        self.client = oandapyV20.API(access_token=self.api_key)
    
        self.min_window_size = 14  # Adjusted for typical technical indicators (e.g., RSI)
        # self.window_size = None

    # def create_sliding_window_dataset(self, data, window_size):
    #     data_np = data.to_numpy()
    #     if len(data_np) < window_size:
    #         logging.error(f"Not enough data ({len(data_np)}) for window size {window_size}")
    #         return np.array([])
    #     windows = np.lib.stride_tricks.sliding_window_view(data_np, window_shape=(window_size,), axis=0)
    #     return windows.squeeze().transpose(0, 2, 1)
        
    def get_data(self, instrument, start_date, end_date, granularity, window_size):
        self.window_size = window_size
        start_date_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_date_dt = pd.to_datetime(end_date).tz_localize('UTC')
        fetch_start = start_date_dt - timedelta(days=self.min_window_size * 2)

        data = pd.DataFrame()
        current_start_date = fetch_start
        max_count = 5000

        while current_start_date < end_date_dt:
            tasks_params = {
                "granularity": granularity,
                "instrument": instrument,
                "from": current_start_date.isoformat(),
                "count": max_count
            }
            df = fetch_chunk(tasks_params, self.client)
            if df.empty:
                break
            last_time = df['time'].max()
            if pd.isnull(last_time):
                break
            current_start = last_time + pd.Timedelta(minutes=1)
            data = pd.concat([data, df], ignore_index=True)
            current_start_date = current_start
        
        if data.empty:
            logging.error('No data fetched')
            return np.array([]), []
        
        if len(data) < self.min_window_size:
            logging.error(f'Insufficient data points ({len(data)}) for technical analysis. Need at least {self.min_window_size} points.')
            return np.array([]), []
        
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)
        data = data.ffill().dropna()

        try:
            # Configure default window sizes for technical indicators
            data = add_all_ta_features(
                data, 
                open='open', 
                high='high', 
                low='low', 
                close='close', 
                volume='volume',
                fillna=True,
            )
            data = data.loc[:end_date_dt]

            if len(data) < window_size:
                logging.error(f"Need {window_size} points, got {len(data)} after trimming")
                return np.array([]), []

            self.feature_names = data.columns.tolist()
            # windows = self.create_sliding_window_dataset(data, self.window_size)
            data_array = data.values
            logging.info(f"Data fetched and processed for {instrument} from {start_date} to {end_date}")
            return data_array, self.feature_names
        except Exception as e:
            logging.error(f'Error calculating technical analysis: {e}')
            return np.array([]), []