import oandapyV20
from datetime import datetime, timedelta
import time
import os
import json
import requests
import backtrader as bt
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
    
def fetch_historical_data(config, instrument, start, end, granularity):
    """
    Fetch historical data from the OANDA API in manageable chunks.
    """
    headers = {"Authorization": f"Bearer {config['access_token']}"}
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    max_candles = 5000  # Maximum candles per request

    # Determine the time delta per candle based on granularity
    granularity_map = {
        'S5': timedelta(seconds=5),
        'S10': timedelta(seconds=10),
        'S15': timedelta(seconds=15),
        'S30': timedelta(seconds=30),
        'M1': timedelta(minutes=1),
        'M2': timedelta(minutes=2),
        'M4': timedelta(minutes=4),
        'M5': timedelta(minutes=5),
        'M10': timedelta(minutes=10),
        'M15': timedelta(minutes=15),
        'M30': timedelta(minutes=30),
        'H1': timedelta(hours=1),
        'H2': timedelta(hours=2),
        'H3': timedelta(hours=3),
        'H4': timedelta(hours=4),
        'H6': timedelta(hours=6),
        'H8': timedelta(hours=8),
        'H12': timedelta(hours=12),
        'D': timedelta(days=1),
        'W': timedelta(weeks=1),
        'M': timedelta(days=30),  # Approximation for a month
    }

    if granularity not in granularity_map:
        raise ValueError(f"Unsupported granularity: {granularity}")

    candle_duration = granularity_map[granularity]
    max_duration = candle_duration * max_candles

    all_candles = []
    next_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))

    while next_start < end_time:
        next_end = min(next_start + max_duration, end_time)
        params = {
            "from": next_start.isoformat(),
            "to": next_end.isoformat(),
            "granularity": granularity,
            "price": "M"  # Midpoint prices
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            candles = response.json().get("candles", [])
            all_candles.extend(candles)

            if not candles or candles[-1]["time"] >= end:
                break  # Exit loop if no more data or reached the end time

            # Update next_start to fetch the next chunk
            next_start = datetime.fromisoformat(candles[-1]["time"].replace('Z', '+00:00'))
        else:
            print(f"Error fetching historical data: {response.json()}")
            break

    print(f"Fetched {len(all_candles)} candles successfully.")
    return all_candles

    
def format_data_for_backtrader(candles):
    """
    Convert OANDA candle data into a Pandas DataFrame suitable for Backtrader.
    """
    import pandas as pd
    data = []
    for candle in candles:
        data.append({
            "datetime": candle["time"],
            "open": float(candle["mid"]["o"]),
            "high": float(candle["mid"]["h"]),
            "low": float(candle["mid"]["l"]),
            "close": float(candle["mid"]["c"]),
            "volume": int(candle["volume"]),
        })
    return pd.DataFrame(data)

def backtesting_strategy(dataframe):
    """
    Run a simple backtesting strategy using Backtrader.
    """
    class MyStrategy(bt.Strategy):
        def __init__(self):
            self.data_close = self.datas[0].close
            
        def next(self):
            if not self.position:
                if self.data_close[0] > self.data_close[-1]:
                    self.buy(size=1)
            else:
                if self.data_close[0] < self.data_close[-1]:
                    self.sell(size=1)
            
    # Initialize Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MyStrategy)
    
    # Load data into Backtrader
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)
    
    # Run backtesting
    console.print("[green]Starting backtesting...[/green]")
    cerebro.run()
    console.print("[blue]Backtesting complete. Generating report...[/blue]")
    cerebro.plot()
    

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
    candles = fetch_historical_data(config, "EUR_USD", "2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z", "H1")
    if candles:
        df = format_data_for_backtrader(candles)
        backtesting_strategy(df)
   
    
    
    
    
    