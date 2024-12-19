import oandapyV20
import time
import os
import json
import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live

CONFIG_FILE = "config.json"
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/accounts"
console = Console()

def load_config():
    """
    Load the configuration file if it exists, otherwise create one.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    else:
        return create_config()
    
def create_config():
    """
    Prompt the user to create a new configuration file.
    """
    print(f"{CONFIG_FILE} not found or credentials invalid. Let's create a new one.")
    account_id = input("Enter your OANDA Account ID: ").strip()
    access_token = input("Enter your OANDA Acess Token: ").strip()
    
    config = {
        "account_id": account_id,
        "access_token": access_token
        }
    
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=4)
        print(f"Configuration saved to {CONFIG_FILE}.")
        
    return config

def validate_config(config):
    """
    Validate the provided account ID and access token with the OANDA API.
    """
    headers={
        "Authorization": f"Bearer {config['access_token']}"
    }
    try:
        print("Validating your credentials...")
        response = requests.get(f"{OANDA_API_URL}/{config['account_id']}", headers=headers)
        if response.status_code == 200:
            print("Credentials validated successfully!")
            return True
        else:
            print("Validation failed. Please check your account ID and account access token.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while validating credentials: {e}")
        return False
    
def get_account_details(config):
    """
    Fetch and display account details using the OANDA API.
    """
    headers = {
        "Authorization": f"Bearer {config['access_token']}"
    }
    try:
        print("Fetching account details...")
        response = requests.get(f"{OANDA_API_URL}/{config['account_id']}", headers=headers)
        if response.status_code == 200:
            account_details = response.json()
            print("Account summary successfully fetched.")
            return account_details
        else:
            print(f"Failed to fetch account details: {response.status_code} - {response.json().get('errorMessage', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching account details: {e}")
        return None
        
def parse_account_details(account_details):
    """
    Parse the account details for relevant information.
    """
    print("Parsing account details...")

    # Extract relevant data
    balance_info = {
        "balance": account_details.get("account", {}).get("balance"),
        "unrealizedPL": account_details.get("account", {}).get("unrealizedPL"),
        "marginUsed": account_details.get("account", {}).get("marginUsed"),
        "marginAvailable": account_details.get("account", {}).get("marginAvailable"),
    }

    open_trades = account_details.get("account", {}).get("trades", [])

    print("Parsed account details successfully.")
    return balance_info, open_trades

def trade_tracking_table(balance_info, open_trades):
    """
    Create a rich table for trade tracking information.
    """
    table = Table(title="Trade Tracking System", expand=True)
    
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    
    # Balance Information
    table.add_row("Balance", balance_info.get("balance", "N/A"))
    table.add_row("Unrealized P/L", balance_info.get("unrealizedPL", "N/A"))
    table.add_row("Margin Used", balance_info.get("marginUsed", "N/A"))
    table.add_row("Margin Available", balance_info.get("marginAvailable", "N/A"))
    
    table.add_section()
    
    # Open Trades
    table.add_column("Open Trades", justify="left", style="cyan", no_wrap=True, header_style="White")
    if open_trades:
        for trade in open_trades:
            table.add_row(f"Instrument: {trade['instrument']}")
            table.add_row(f"Units: {trade['currentUnits']}")
            table.add_row(f"Open Price: {trade['price']}")
            table.add_row(f"Unrealized P/L: {trade['unrealizedPL']}")
    else:
        table.add_row("No open trades.")
    return table

def update_live_display(config):
    """
    Continuously update the live display with account details.
    """
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            account_details = get_account_details(config)
            if account_details:
                balance_info, open_trades = parse_account_details(account_details)
                table = trade_tracking_table(balance_info, open_trades)
                live.update(table)
            time.sleep(5)
        
if __name__ == "__main__":
    config = load_config()
    while not validate_config(config):
        print("Your credentials seem to be invalid. Let's try again.")
        config = create_config()
    print("Your configuration:", config)
    account_raw = get_account_details(config)
    balance_info, open_trades = parse_account_details(account_raw)
    trade_tracking_table(balance_info, open_trades)
    while True:
        update_live_display(config)
   
    
    
    
    
    