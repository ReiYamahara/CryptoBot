from mean_reversion import *
import os
import ccxt
from dotenv import load_dotenv
import pandas as pd
import time
import math
from state_manager import StateManager

state_manager = StateManager()

SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
WINDOW_SIZE = 64

load_dotenv()

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_PRIVATE_KEY = os.getenv("KRAKEN_PRIVATE_KEY")

# 1. SETUP KRAKEN CONNECTION
exchange = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_PRIVATE_KEY,
    'enableRateLimit': True,
})

strategies = [MeanReversionStrategy()]
current_positions = [0.0]*len(strategies)

def fetch_live_data(symbol, timeframe, limit=100):
    """
    Fetches OHLCV data from Kraken
    """
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def wait_for_next_minute():
    """
    Sleeps exactly until the start of the next minute + 1 second buffer.
    """
    now = time.time()
    # Calculate time to next full minute
    next_minute = math.ceil(now / 60) * 60
    
    # Add 1-2 seconds buffer to ensure Kraken has finished writing the candle
    sleep_seconds = next_minute - now + 2 
    
    if sleep_seconds < 0:
        sleep_seconds += 60
        
    print(f"Waiting {sleep_seconds:.2f}s for next candle...")
    time.sleep(sleep_seconds)

while True:
    data = fetch_live_data(symbol=SYMBOL, timeframe=TIMEFRAME, limit=100)

    for strategy in strategies:
        signal = strategy.on_data(data)
        name = "PLACEHOLDER" # ADD A STRATEGY.GETNAME() FUNCTION FOR EACH NEW STRATEGY
            
        market_order = exchange.create_market_buy_order('BTC/USD', 0.01)
        print("Market Order Successful:", market_order['id'])
    pass