import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.key_handler import decrypt_data, load_key
from src.data_handler import DataHandler
import pandas as pd
from unittest.mock import patch

class TestDataHandlerFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the key to ensure encryption setup
        load_key()
        
        # Setup encryption and save the key and encrypted data
        cls.api_key = 'test_api_key'
        cls.account_id = 'test_account_id'
        cls.data_handler = DataHandler()

    @patch('data_handler.DataHandler.get_data')
    def test_data_fetching(self, mock_get_data):
        # Mock the get_data method to return a sample dataframe
        mock_get_data.return_value = (pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.2, 1.3, 1.4, 1.5, 1.6],
            'low': [1.0, 1.1, 1.2, 1.3, 1.4],
            'close': [1.15, 1.25, 1.35, 1.45, 1.55],
            'volume': [100, 200, 300, 400, 500]
        }), ['time', 'open', 'high', 'low', 'close', 'volume'])

        # Fetch data using the DataHandler
        instrument = 'EUR_USD'
        start_date = '2023-01-01'
        end_date = '2023-01-05'
        granularity = 'D'
        data, _ = self.data_handler.get_data(instrument, start_date, end_date, granularity, window_size=5)

        # Verify the fetched data
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 5)

if __name__ == '__main__':
    unittest.main()