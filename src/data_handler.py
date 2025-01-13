import oandapyV20
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from oandapyV20.endpoints.instruments import InstrumentsCandles
import logging
from ta import add_all_ta_features
from .key_handler import decrypt_data  # Updated import statement

def fetch_chunk(params, client):
    try:
        response = client.request(InstrumentsCandles(instrument=params['instrument'], params=params))
        candles = response.get('candles', [])
        if len(candles) < 5000 or len(candles) == 0:
            logging.warning(f'EOD reached for {params["from"]}')
        df = pd.DataFrame([{
            'time': candle['time'],
            'open': candle['mid']['o'],
            'high': candle['mid']['h'],
            'low': candle['mid']['c'],
            'close': candle['mid']['c'],
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
        
    def get_data(self, instrument, start_date, end_date, granularity):
        params = {
            "granularity": granularity,
            "count": 5000,
            "instrument": instrument
        }
        data = pd.DataFrame()
        current_start_date = datetime.fromisoformat(start_date)
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
            return data

        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)

        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)

        data = add_all_ta_features(data, open='open', high='high', low='low', close='close', volume='volume')

        logging.info(f"Data fetched and processed for {instrument} from {start_date} to {end_date}")

        return data