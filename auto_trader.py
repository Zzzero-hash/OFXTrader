import sys
from src.config_utils import load_config, validate_config
from src.server import run_server
from src.account_details import update_live_display
# from src.data_fetcher import fetch_historical_data (if you want to call from main)
# from src.oanda_client import fetch_instrument_list
# ... etc.

def main():
    """
    Main entry point for your trading bot.
    """
    config = load_config()
    if not validate_config(config):
        print("Exiting due to invalid credentials.")
        sys.exit(1)

    print("Starting trading bot...")

    # Example usage: Start live display in a separate thread or process
    # Or just run the server if you want the webhook to handle orders.
    # update_live_display(config)

    # Start the Flask server to receive webhook signals
    run_server()

if __name__ == "__main__":
    main()
