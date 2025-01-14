import sys
import os
import logging
import unittest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.key_handler import decrypt_data, setup_encryption
from src.data_handler import DataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Decrypt the data to ensure it was stored correctly
    print("Decrypting data to verify...")
    api_key, account_id = decrypt_data()
    print(f"Decrypted API Key: {api_key}")
    print(f"Decrypted Account ID: {account_id}")
    
    # Step 2: Initialize DataHandler with the decrypted data
    print("Initializing DataHandler...")
    data_handler = DataHandler()
    
    try:
        # Step 3: Fetch data using DataHandler
        print("Fetching data...")
        instrument = 'EUR_USD'
        start_date = '2023-01-01'
        end_date = '2023-10-01'
        granularity = 'M5'
        data = data_handler.get_data(instrument, start_date, end_date, granularity)
        
        # Step 4: Display the fetched data
        print("Fetched data:")
        print(data.head())
    except Exception as e:
        # Log the error
        logging.error(f"Error fetching data: {e}")

class TestDataHandler(unittest.TestCase):
    def test_get_data(self):
        data_handler = DataHandler()
        instrument = 'EUR_USD'
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        granularity = 'D'
        data = data_handler.get_data(instrument, start_date, end_date, granularity)
        
        # Check that data is not None and not empty
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        
        # Check that the expected columns are present
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in expected_columns:
            self.assertIn(column, data.columns)
        
        # Check that the data types are correct
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data.index))
        self.assertTrue(pd.api.types.is_float_dtype(data['open']))
        self.assertTrue(pd.api.types.is_float_dtype(data['high']))
        self.assertTrue(pd.api.types.is_float_dtype(data['low']))
        self.assertTrue(pd.api.types.is_float_dtype(data['close']))
        self.assertTrue(pd.api.types.is_integer_dtype(data['volume']))
        
        # Check that there are no NaN values
        self.assertFalse(data.isnull().values.any())
        
        # Check that the data covers the expected date range
        start_date_tz_aware = pd.to_datetime(start_date).tz_localize(data.index.tz)
        end_date_tz_aware = pd.to_datetime(end_date).tz_localize(data.index.tz)
        self.assertGreaterEqual(data.index.min(), start_date_tz_aware)
        self.assertLessEqual(data.index.max(), end_date_tz_aware)

if __name__ == '__main__':
    main()
    unittest.main()