import warnings
import datetime
import logging
import os
from config_utils import load_or_create_config
from train import initial_setup  # Retain only training utilities if needed

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
DATE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

# Example usage
if __name__ == "__main__":
    config = load_or_create_config()
    import argparse
    parser = argparse.ArgumentParser()

    start_date = (datetime.datetime.now() - datetime.timedelta(days=3650)).strftime(DATE_FORMAT)
    end_date = datetime.datetime.now().strftime(DATE_FORMAT)
    parser.add_argument('--granularity', type=str, required=True, help="Granularity (e.g., 'H1')")
    parser.add_argument('--instrument', type=str, required=True, help="Instrument (e.g., 'EUR_USD')")
    parser.add_argument('--max-profit-percent', type=float, default=0.1, help="Stop after reaching this profit percentage")
    parser.add_argument('--max-loss-percent', type=float, default=0.1, help="Stop after reaching this loss percentage")
    args = parser.parse_args()

    instrument = args.instrument
    granularity = args.granularity
    start_date = (datetime.datetime.now() - datetime.timedelta(days=3650)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    access_token = config["access_token"]
    account_id = config["account_id"]
    
    available_models = os.listdir("trained_models") if os.path.exists("trained_models") else []
    print("Available pre-trained models (if any):")
    if available_models:
        for model in available_models:
            print(f"- {model}")
    
    model_name = input("Enter the name of the pre-trained model to load (leave blank to train a new model): ")
    
    initial_setup()
    try:
        if model_name:
            # Import and initialize the TradingBot for live trading
            from trade import TradingBot
            bot = TradingBot(
                model_name=model_name,
                instrument=instrument,
                granularity=granularity,
                start_date=start_date,
                end_date=end_date,
                access_token=access_token,
                max_profit_percent=args.max_profit_percent,
                max_loss_percent=args.max_loss_percent
            )
            bot.start_live_trading()
        else:
            # Handle training if no pre-trained model is selected
            from train import train_model
            # Fetch and preprocess data as needed
            # ...existing training code...
    except Exception as e:
        logging.error(f"An error occurred while running the bot: {e}")
        print(f"An error occurred while running the bot: {e}")
