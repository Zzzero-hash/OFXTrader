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
            instrument = params.pop('instrument')  # Correctly pop the instrument from params
            logging.info(f"Fetching data with params: {params}")
            response = client.request(InstrumentsCandles(instrument=instrument, params=params))
            logging.info(f"Response received: {response}")
            candles = response.get('candles', [])
            if len(candles) == 0:
                logging.warning(f'EOD reached for {params["from"]}')
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
        except Exception as e:
            logging.warning(f"Retry {_+1}/3 failed: {e}")
            time.sleep(2)
    return pd.DataFrame()

class DataHandler:
    def __init__(self):
        api_key, account = decrypt_data()
        self.api_key = api_key
        self.account = account
        self.client = oandapyV20.API(access_token=api_key)
        self.min_window_size = 14  # Adjusted for typical technical indicators (e.g., RSI)
        self.window_size = None

    def create_sliding_window_dataset(self, data, window_size):
        data_np = data.to_numpy()
        if len(data_np) < window_size:
            logging.error(f"Not enough data ({len(data_np)}) for window size {window_size}")
            return np.array([])
        windows = np.lib.stride_tricks.sliding_window_view(data_np, window_shape=(window_size,), axis=0)
        return windows.squeeze().transpose(0, 2, 1)
        
    def get_data(self, instrument, start_date, end_date, granularity, window_size):
        self.window_size = window_size
        params = {
            "granularity": granularity,
            "instrument": instrument
        }
        start_date_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_date_dt = pd.to_datetime(end_date).tz_localize('UTC')

        fetch_start = start_date_dt - timedelta(days=self.min_window_size * 2)

        data = pd.DataFrame()
        
        # Adjust start_date to fetch extra data for technical analysis window
        current_start_date = fetch_start
        tasks = []

        while current_start_date < end_date_dt:
            current_end_date = current_start_date + timedelta(days=5)
            if current_end_date > end_date_dt:
                current_end_date = end_date_dt
            task_params = params.copy()
            task_params["from"] = current_start_date.isoformat()
            task_params["to"] = current_end_date.isoformat()
            tasks.append((task_params, self.client))
            current_start_date = current_end_date

        with Pool(processes=4) as pool:
            results = pool.starmap(fetch_chunk, tasks)
            data = pd.concat(results, ignore_index=True)

        if data.empty:
            logging.error('No data fetched')
            return np.array([]), []

        if len(data) < self.min_window_size:
            logging.error(f'Insufficient data points ({len(data)}) for technical analysis. Need at least {self.min_window_size} points.')
            return []

        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)

        data.ffill().dropna()

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
            windows = self.create_sliding_window_dataset(data, self.window_size)
            
        except Exception as e:
            logging.error(f'Error calculating technical analysis: {e}')
            return np.array([]), []
        
        logging.info(f"Data fetched and processed for {instrument} from {start_date} to {end_date}")
        return windows, self.feature_names