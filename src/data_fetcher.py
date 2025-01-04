import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser

import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles

from .oanda_client import get_oanda_client

def fetch_historical_data(config, instrument, start=None, end=None, granularity="M15"):
    """
    Fetch historical candle data from OANDA, handling API limits for maximum candles.
    """
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

    # Parse dates
    try:
        start_dt = parser.parse(start).replace(tzinfo=None)
        end_dt = parser.parse(end).replace(tzinfo=None)
    except (TypeError, ValueError) as e:
        print(f"Error parsing dates: {e}")
        return pd.DataFrame()

    max_candles = 5000  # OANDA's maximum count limit
    all_data = []

    # Granularity in seconds
    granularity_to_seconds = {
        "S5": 5, "S10": 10, "S15": 15, "S30": 30,
        "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600,
        "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200,
        "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800,
        "H12": 43200, "D": 86400, "W": 604800, "M": 2592000
    }
    granularity_seconds = granularity_to_seconds.get(granularity, 3600)

    current_start = start_dt
    while current_start < end_dt:
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

        for candle in candles:
            all_data.append({
                "datetime": candle["time"],
                "open": float(candle["mid"]["o"]),
                "high": float(candle["mid"]["h"]),
                "low": float(candle["mid"]["l"]),
                "close": float(candle["mid"]["c"]),
                "volume": int(candle["volume"]),
            })

        # Move to the next chunk
        last_candle_time = parser.parse(candles[-1]["time"]).replace(tzinfo=None)
        current_start = last_candle_time + timedelta(seconds=granularity_seconds)

        print(f"Fetched {len(candles)} candles. Total so far: {len(all_data)}")

    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    else:
        print("No data returned for the specified range.")
        df = pd.DataFrame()

    return df