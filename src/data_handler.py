import oandapyV20
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from oandapyV20.endpoints.instruments import InstrumentsCandles
import logging
from ta import add_all_ta_features
from .key_handler import decrypt_data  # Updated import statement

def fetch_chunk(params, client):
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
        return df
    except Exception as e:
        logging.error(f'Error fetching data: {e}')
        return pd.DataFrame()

class DataHandler:
    def __init__(self):
        api_key, account = decrypt_data()
        self.api_key = api_key
        self.account = account
        self.client = oandapyV20.API(access_token=api_key)
        self.min_window_size = 14  # Adjusted for typical technical indicators (e.g., RSI)
        self.window_size = None

    def create_windowed_dataset(self, data, window_size):
        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data.iloc[i:i + window_size].values)
        return windows
        
    def get_data(self, instrument, start_date, end_date, granularity, window_size):
        params = {
            "granularity": granularity,
            "instrument": instrument
        }
        data = pd.DataFrame()
        
        # Adjust start_date to fetch extra data for technical analysis window
        current_start_date = datetime.fromisoformat(start_date) - timedelta(days=self.min_window_size * 2)  # Double the window for safety
        end_date = datetime.fromisoformat(end_date)
        tasks = []

        while current_start_date < end_date:
            current_end_date = current_start_date + timedelta(days=5)
            if current_end_date > end_date:
                current_end_date = end_date
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
            return []

        # Remove duplicate rows that might come from overlapping date ranges
        data = data.loc[~data.index.duplicated(keep='first')]
        data = data.sort_index()

        if len(data) < self.min_window_size:
            logging.error(f'Insufficient data points ({len(data)}) for technical analysis. Need at least {self.min_window_size} points.')
            return []

        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)

        data.ffill(inplace=True)
        data.dropna(inplace=True)

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
            
            # Trim the extra data we fetched for the window
            data = data[start_date:]

            data = self.create_windowed_dataset(data, self.min_window_size)
            
        except Exception as e:
            logging.error(f'Error calculating technical analysis: {e}')
            return []

        logging.info(f"Data fetched and processed for {instrument} from {start_date} to {end_date}")
        return data