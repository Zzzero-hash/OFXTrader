from functools import lru_cache
import pandas as pd
import multiprocessing
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from concurrent.futures import ThreadPoolExecutor
import ta

@lru_cache(maxsize=32)
def fetch_oanda_candles_chunk(instrument, from_time, to_time, granularity, access_token):
    """Fetch up to `count` candles in one request."""
    client = oandapyV20.API(access_token=access_token)
    params = {
        "from": pd.Timestamp(from_time).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "to": pd.Timestamp(to_time).strftime('%Y-%m-%dT%H:%M:%SZ'),      
        "granularity": granularity
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    resp = client.request(r)
    
    # Parse into DataFrame
    data = resp["candles"]
    df = pd.DataFrame([{
        'time': candle['time'],
        'open': float(candle['mid']['o']),
        'high': float(candle['mid']['h']),
        'low': float(candle['mid']['l']),
        'close': float(candle['mid']['c']),
        'volume': int(candle['volume'])
    } for candle in data])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    return df

def fetch_oanda_candles_range(instrument, start_date, end_date, granularity, access_token):
    current_start = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Calculate chunks for parallel processing
    chunks = []
    while current_start < end_date:
        to_dt = min(current_start + pd.Timedelta(minutes=5000), end_date)
        chunks.append((current_start, to_dt))
        current_start = to_dt

    # Parallel fetch chunks
    max_workers = min(len(chunks), multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for start, to_dt in chunks:
            futures.append(executor.submit(
                fetch_oanda_candles_chunk,
                instrument,
                start,
                to_dt,
                granularity,
                access_token
            ))
        results = [f.result() for f in futures]

    return pd.concat(results, ignore_index=True)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    
    # Vectorized calculations
    df['rsi_14'] = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    df['dx_14'] = ta.trend.ADXIndicator(high=high_prices, low=low_prices, close=close_prices, window=14).adx()
    
    # Optimize Bollinger Bands calculation
    bb = ta.volatility.BollingerBands(close=close_prices, window=14)
    df['bb_ub'] = bb.bollinger_hband()
    df['bb_lb'] = bb.bollinger_lband()
    
    # Use inplace operations
    df.dropna(inplace=True)
    return df