import oandapyV20
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from oandapyV20.endpoints.instruments import InstrumentsCandles
import logging
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from itertools import combinations
from key_handler import decrypt_data  # Updated import statement

def fetch_chunk(params, client):
    for attempt in range(3):  # Retry fetching data up to 3 times
        try:
            request_params = params.copy()
            instrument = request_params.pop('instrument')
            response = client.request(InstrumentsCandles(instrument=instrument, params=request_params))
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
        """Initialize the DataHandler with an OANDA API client."""
        self.api_key = '7d72ad59524a9f896c85eb7cc9d21a37-aa2a02a5157b1891340ac08e9a1d1c29'
        self.account = '101-001-23675199-001'
        self.client = oandapyV20.API(access_token=self.api_key)
        
    def _get_instrument_data(self, instrument, start_date, end_date, granularity):
        """
        Fetch and process data for a single instrument, including selected technical indicators.
        
        Args:
            instrument (str): The Forex instrument (e.g., 'EUR_USD').
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            granularity (str): Data granularity (e.g., 'M1', 'D').
        
        Returns:
            pd.DataFrame: Processed dataframe with OHLCV and technical indicators.
        """
        data = pd.DataFrame()
        current_start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date_dt = pd.to_datetime(end_date).tz_localize('UTC')
        max_count = 5000

        # Fetch data in chunks
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
            logging.error(f'No data fetched for {instrument}')
            return np.array([]), []
        
        # Set time index and clean data
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)
        data = data.ffill().dropna()

        # Compute selected technical indicators
        data['RSI_14'] = RSIIndicator(data['close'], window=14).rsi()
        macd = MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
        data['MA_200'] = data['close'].rolling(window=200).mean()
        data['ATR_14'] = AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()

        # Selected desired columns
        selected_columns = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'MA_200', 'ATR_14']
        data = data[selected_columns]
        return data
            
    def get_data(self, instruments, start_date, end_date, granularity, window_size, correlation_window_days=30):
        """
        Fetch and process data for multiple instruments, including cross-market correlation features.
        
        Args:
            instruments (list): List of Forex instruments (e.g., ['EUR_USD', 'USD_JPY']).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            granularity (str): Data granularity (e.g., 'M1').
            window_size (int): Window size for RL observation sequences (not used in data prep here).
            correlation_window_days (int): Window in days for rolling correlation (default: 30).
        
        Returns:
            tuple: (numpy array of data, list of feature names)
        """
        # Fetch daily data for correlations
        daily_start_date_dt = pd.to_datetime(start_date) - pd.Timedelta(days=correlation_window_days)
        daily_data = {}
        for instrument in instruments:
            df_daily = self._get_instrument_data(instrument, daily_start_date_dt.strftime('%Y-%m-%d'), end_date, 'D')
            if not df_daily.empty:
                daily_data[instrument] = df_daily['close']

        if not daily_data:
            logging.error('No daily data fetched for correlation features.')
            return np.array([]), []
        
        # Concatenate daily close prices
        df_daily_all = pd.concat([daily_data[instrument].rename(f'close_{instrument}') for instrument in instruments], axis=1)

        # Compute rolling correlations for each pair
        correlation_features = {}
        for instr1, instr2 in combinations(instruments, 2):
            corr_series = df_daily_all[f'close_{instr1}'].rolling(window=correlation_window_days).corr(df_daily_all[f'close_{instr2}'])
            correlation_features[f'correlation_{instr1}_{instr2}'] = corr_series.values
        df_correlations = pd.DataFrame(correlation_features, index=df_daily_all.index)

        # Fetch data for each instrument
        main_data = {}
        for instrument in instruments:
            df = self._get_instrument_data(instrument, start_date, end_date, granularity)
            if not df.empty:
                main_data[instrument] = df

        if not main_data:
            logging.error('No main data fetched for selected instruments.')
            return np.array([]), []
        
        # Concatenate main data on time index
        df_main = pd.concat(main_data.values(), axis=1)

        # Add date column for merging correlations
        df_main['date'] = df_main.index.date
        
        # Create date column in correlations dataframe
        df_correlations['date'] = df_correlations.index.date
        
        # Merge daily correlations into main data
        df_main = df_main.merge(df_correlations, on='date', how='left')

        # Drop date column and remove rows with NaN values
        df_main = df_main.drop(columns=['date']).dropna()

        # Convert to numpy array and get feature names
        data_array = df_main.values
        feature_names = df_main.columns.tolist()

        logging.info(f"Data `{instruments}` fetched successfully. Shape: {data_array.shape}, Features: {feature_names}")
        return data_array, feature_names