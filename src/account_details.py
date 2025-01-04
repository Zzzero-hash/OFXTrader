import time
import requests
from rich.console import Console
from rich.live import Live
from .trade_tracking import trade_tracking_table

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/accounts"
console = Console()

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

    balance_info = {
        "balance": account_details.get("account", {}).get("balance"),
        "unrealizedPL": account_details.get("account", {}).get("unrealizedPL"),
        "marginUsed": account_details.get("account", {}).get("marginUsed"),
        "marginAvailable": account_details.get("account", {}).get("marginAvailable"),
    }

    open_trades = account_details.get("account", {}).get("trades", [])
    print("Parsed account details successfully.")

    return balance_info, open_trades

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
