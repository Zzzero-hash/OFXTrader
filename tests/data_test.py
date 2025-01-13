import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.key_handler import setup_encryption, decrypt_data
from src.data_handler import DataHandler

def main():
    # Step 1: Setup encryption by asking the user for API key and account ID
    print("Setting up encryption...")
    setup_encryption()
    
    # Step 2: Decrypt the data to ensure it was stored correctly
    print("Decrypting data to verify...")
    api_key, account_id = decrypt_data()
    print(f"Decrypted API Key: {api_key}")
    print(f"Decrypted Account ID: {account_id}")
    
    # Step 3: Initialize DataHandler with the decrypted data
    print("Initializing DataHandler...")
    data_handler = DataHandler()
    
    # Step 4: Fetch data using DataHandler
    print("Fetching data...")
    instrument = 'EUR_USD'
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    granularity = 'D'
    data = data_handler.get_data(instrument, start_date, end_date, granularity)
    
    # Step 5: Display the fetched data
    print("Fetched data:")
    print(data.head())

if __name__ == '__main__':
    main()