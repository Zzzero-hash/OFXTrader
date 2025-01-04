import os
import json
import requests

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
CONFIG_FILE = os.path.join("config", "config.json")

def load_config():
    """Load the config file otherwise create a new one and save it to the config folder."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    else:
        return create_config()

def create_config():
    """
    Prompt the user to enter their account ID and API key.
    Save the config to the config folder.
    """
    print(f"{CONFIG_FILE} not found or credentials were invalid. Please enter your account ID and API key.")

    account_id = input("Account ID: ").strip()
    api_key = input("API Key: ").strip()
    config = {"account_id": account_id, "access_token": api_key}

    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=4)
        print(f"Config saved to {CONFIG_FILE}")

    return config

    def validate_config():
        """
        Validate the provided account ID and access token with the OANDA API.
        If the credentials are invalid, prompt the user to enter them again.
        """
        config = load_config()
        headers = {"Authorization": f"Bearer {config['access_token']}"}

        try:
            print("Validating credentials...")
            response = requests.get(f"{OANDA_API_URL}/accounts/{config['account_id']}", headers=headers)
            if response.status_code == 200:
                print("Credentials validated.")
                return True
            else:
                print("Invalid credentials.")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return False