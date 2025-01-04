import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
from datetime import datetime, timedelta, timezone
import time
import os
import json
import requests
import pandas as pd
from dateutil import parser
from rich.console import Console
from rich.table import Table
from rich.live import Live
import qlib
from qlib.config import REG_US
from qlib.contrib.strategy import TopkDropoutStrategy, Alpha158, Alpha360

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)
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
    access_token = input("Enter your OANDA Access Token: ").strip()
    
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
    
def get_oanda_client(config):
    """
    Create an OANDA API client using the provided configuration.
    """
    return oandapyV20.API(access_token=config["access_token"])


    
def fetch_historical_data(config, instrument, start=None, end=None, granularity=None):
    """
    Fetch historical candle data from OANDA, handling API limits for maximum candles.
    """
    # Validate and set default start and end times
    if not start or not start.strip():
        one_year_ago = datetime.now(tz=timezone.utc) - timedelta(days=365)
        start = one_year_ago.replace(microsecond=0).isoformat()

    if not end or not end.strip():
        today = datetime.now(timezone.utc)
        end = today.replace(microsecond=0).isoformat()

    # Create OANDA API client
    api = get_oanda_client(config)

    # Validate granularity
    supported_granularities = [
        "S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10",
        "M15", "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12",
        "D", "W", "M"
    ]
    if granularity not in supported_granularities:
        raise ValueError(f"Unsupported granularity: {granularity}")

    # Parse dates and ensure they're in the correct format
    try:
        start_dt = parser.parse(start).replace(tzinfo=None)
        end_dt = parser.parse(end).replace(tzinfo=None)
    except (TypeError, ValueError) as e:
        print(f"Error parsing dates: {e}")
        return pd.DataFrame()

    max_candles = 5000  # OANDA's maximum count limit
    all_data = []

    # Calculate the granularity duration in seconds
    granularity_seconds = {
        "S5": 5, "S10": 10, "S15": 15, "S30": 30,
        "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600,
        "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200,
        "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800,
        "H12": 43200, "D": 86400, "W": 604800, "M": 2592000
    }.get(granularity, 3600)

    # Fetch data in chunks
    current_start = start_dt
    while current_start < end_dt:
        # Calculate the chunk end time
        chunk_end = min(
            current_start + timedelta(seconds=max_candles * granularity_seconds),
            end_dt
        )

        params = {
            "from": current_start.isoformat() + 'Z',
            "to": chunk_end.isoformat() + 'Z',
            "granularity": granularity,
            "price": "M",
        }

        print(f"Fetching chunk from {params['from']} to {params['to']}")

        # Fetch data
        r = InstrumentsCandles(instrument=instrument, params=params)
        try:
            response = api.request(r)
        except oandapyV20.exceptions.V20Error as e:
            print(f"V20Error fetching data: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unknown error fetching data: {e}")
            return pd.DataFrame()

        candles = response.get("candles", [])
        if not candles:
            print("No more candle data found.")
            break

        # Append the candles to all_data
        for candle in candles:
            all_data.append({
                "datetime": candle["time"],
                "open": float(candle["mid"]["o"]),
                "high": float(candle["mid"]["h"]),
                "low": float(candle["mid"]["l"]),
                "close": float(candle["mid"]["c"]),
                "volume": int(candle["volume"]),
            })

        # Update start time for the next chunk
        if candles:
            last_candle_time = parser.parse(candles[-1]["time"]).replace(tzinfo=None)
            current_start = last_candle_time + timedelta(seconds=granularity_seconds)
        else:
            break

        print(f"Fetched {len(candles)} candles. Total: {len(all_data)}")

    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        print("No data returned for the specified range.")
        df = pd.DataFrame()

    return df


def backtesting_strategy(dataframe, strategy_name, plot=True):
    # Map strategy names to Qlib strategy implementations
    strategies = {
        "TopkDropoutStrategy": TopkDropoutStrategy(n_drop=5, n_top=10),
        "Alpha158": Alpha158(),
        "Alpha360": Alpha360()
    }

    if strategy_name not in strategies:
        print(f"Unknown strategy: {strategy_name}")
        return
    
    strategy = strategies[strategy_name]

    # Simulate a simple data setup for Qlib (transforming OANDA data for Qlib use)
    df = dataframe.reset_index()
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["symbol"] = "OANDA_SIM"
    df.rename(columns={"close": "feature"}, inplace=True)

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
    table.add_column("Open Trades", justify="left", style="cyan", no_wrap=True)
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

def fetch_instrument_list(config):
    """
    Fetch the list of available instruments for trading.
    """
    headers = {
        "Authorization": f"Bearer {config['access_token']}"
    }
    try:
        print("Fetching instrument list...")
        response = requests.get(f"{OANDA_API_URL}/{config['account_id']}/instruments", headers=headers)
        if response.status_code == 200:
            instruments = response.json().get("instruments", [])
            print("Instrument list successfully fetched.")
            return instruments
        else:
            print(f"Failed to fetch instrument list: {response.status_code} - {response.json().get('errorMessage', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching instrument list: {e}")
        return None
    
def user_manual_bt(config, selected_strategy):
    print(f"Selected strategy: {selected_strategy}")
    candle_instrument = input("Enter the instrument to fetch historical data for (e.g. EUR_USD): ").strip()
    candle_start = input("Enter the start date (e.g. 2023-01-01) or press enter for 1 year ago from today (default): ").strip()
    candle_end = input("Enter the end date (e.g. 2023-12-31) or press enter for now(today): ").strip()
    candle_granularity = input("Enter the granularity (e.g. H1) or press enter for 1 hour candles by default: ").strip()

    # Append time to the date inputs
    if candle_start:
        candle_start += "T00:00:00Z"
    if candle_end:
        candle_end += "T00:00:00Z"

    df = fetch_historical_data(config=config,
                                instrument=candle_instrument, 
                                start=candle_start, 
                                end=candle_end, 
                                granularity=candle_granularity if candle_granularity != '' else "H1"
                                )
    if df.empty:
        print("No valid candle data returned. Aborting backtesting.")
    else:
        print("\nProcessing candles for backtesting...")
        backtesting_strategy(df, selected_strategy)

def user_selection():
    """
    Main user selection menu for the trading bot.
    """
    config = load_config()
    while not validate_config(config):
        print("Your credentials seem to be invalid. Let's try again.")
        config = create_config()
    while True:
        try:
            account_mode = int(input("Enter a mode selection: 1 (Auto) or 2 (Manual): "))
            if account_mode in [1, 2]:
                break # Exit the loop if a valid mode is selected
            else:
                print("Invalid input. Please enter 1 (Auto) or 2 (Manual).")
        except ValueError:
            print("Invalid input. Please enter 1 (Auto) or 2 (Manual).")
            continue
        print(f"Account mode selected: {account_mode}") # Debug print

    # Handle user_mode selection
    while True:
        if account_mode == 1:
            user_mode = int(input("Auto mode selected. Choose bot mode: 1 (Live) or 2 (Backtesting): "))
            if user_mode == 1:
                print(f"User mode selected: {user_mode}") # Debug print
                # TODO: Implement live trading mode
                print("Live trading mode is not implemented yet.")
                print(f"User mode selected: {user_mode}") # Debug print
                print("Backtesting mode is not implemented yet.")
            else:
                print("Invalid mode selection. Please try again.")
                continue
        elif account_mode == 2:
            user_mode = int(input("Manual mode selected. Choose bot mode: 1 (Live) or 2 (Backtesting): "))
            if user_mode == 1:
                print(f"User mode selected: {user_mode}") # Debug print
                print("Manual trading mode is not implemented yet.")
            if user_mode == 2:
                print(f"User mode selected: {user_mode}") # Debug print
                print("Backtesting mode is not implemented yet.")
                print(f"Select an algorithm for backtesting: ")
                strategies = ["TopkDropoutStrategy", "DoubleMaStrategy", "Alpha158", "Alpha360"]
                for i, strategy in enumerate(strategies, 1):
                    print(f"{i}. {strategy}")
                strategy_choice = int(input("Enter the number of the strategy you want to use: "))
                if 1 <= strategy_choice <= len(strategies):
                    selected_strategy = strategies[strategy_choice - 1]
                    print(f"Selected strategy: {selected_strategy}")
                else:
                    print("Invalid selection. Defaulting to MyStrategy.")
                    selected_strategy = "MyStrategy"
                user_manual_bt(config, selected_strategy)
        else:
            print("Invalid mode selection. Please try again.")
            continue

if __name__ == "__main__":
    user_selection()