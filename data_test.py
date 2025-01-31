import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import unittest
import numpy as np
from key_handler import decrypt_data, setup_encryption
from data_handler import DataHandler

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
        end_date = '2024-01-01'
        granularity = 'D'
        data = data_handler.get_data(instrument, start_date, end_date, granularity, window_size=14)
        
        if data[0] is None:
            setup_encryption()
        # Step 4: Display the fetched data
        print("Fetched data:")
        print(data)

    except Exception as e:
        # Log the error
        logging.error(f"Error fetching data: {e}")

class TestDataHandler(unittest.TestCase):
    def test_get_data(self):
        data_handler = DataHandler()
        instrument = 'EUR_USD'
        start_date = '2023-01-01'
        end_date = '2023-12-01'
        granularity = 'D'
        windows, features = data_handler.get_data(instrument, start_date, end_date, granularity, window_size=14)
        self.assertIsInstance(windows, np.ndarray)
        self.assertIsInstance(features, list)
        self.assertGreater(windows.shape[0], 0) # Check number of windows
        self.assertEqual(windows.shape[2], len(features)) # Features should match

if __name__ == '__main__':
    main()
    unittest.main()