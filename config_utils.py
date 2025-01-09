import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import logging
import json
import os
from typing import Dict
logging.basicConfig(level=logging.INFO)

def load_or_create_config(config_path: str = "config.json") -> Dict[str, str]:
    """Load the config file if it exists, otherwise create it by prompting the user for credentials."""
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
            if validate_credentials(config["access_token"], config["account_id"]):
                return config
            else:
                logging.error("Stored credentials are invalid. Please re-enter your credentials.")
    
    # Prompt the user for credentials if the file does not exist or credentials are invalid
    while True:
        config = get_credentials_from_user()
        if validate_credentials(config["access_token"], config["account_id"]):
            with open(config_path, "w") as file:
                json.dump(config, file)
            return config
        else:
            logging.error("Invalid credentials. Please try again.")

def get_credentials_from_user() -> Dict[str, str]:
    """Prompt the user to enter their OANDA account ID and access token."""
    account_id = input("Enter your OANDA account ID: ")
    access_token = input("Enter your OANDA access token: ")
    return {"account_id": account_id, "access_token": access_token}

def validate_credentials(access_token: str, account_id: str) -> bool:
    """Validate the provided OANDA credentials."""
    try:
        client = oandapyV20.API(access_token=access_token)
        r = accounts.AccountDetails(account_id)  # Use the correct import
        client.request(r)
        return True
    except Exception as e:
        logging.error(f"Invalid credentials: {e}")
        return False