from mean_reversion import MeanReversionStrategy
import os
import ccxt
from dotenv import load_dotenv
import pandas as pd
import time
import math
from state_manager import StateManager

# --- CONFIGURATION ---
SYMBOL = 'BTC/USD'
TIMEFRAME = '1m'
COIN = 'BTC'
QUOTE = 'USD'

# Load Environment
load_dotenv()
state_manager = StateManager()

exchange = ccxt.kraken({
    'apiKey': os.getenv("KRAKEN_API_KEY"),
    'secret': os.getenv("KRAKEN_PRIVATE_KEY"),
    'enableRateLimit': True,
})

# --- SETUP STRATEGIES ---
# We use a List of Dictionaries for easier updates
strategy_configs = [
    {
        "strategy": MeanReversionStrategy(),
        "name": "Mean Reversion",
        "allocation_pct": 0.2  # 20% of total USDT balance
    }
    # Add your VAE strategy here later
]

def fetch_live_data(symbol, timeframe, limit=100):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def wait_for_next_minute():
    now = time.time()
    next_minute = math.ceil(now / 60) * 60
    sleep_seconds = next_minute - now + 2 
    if sleep_seconds < 0: sleep_seconds += 60
    print(f"Waiting {sleep_seconds:.2f}s for next candle...")
    time.sleep(sleep_seconds)

def get_portfolio_value():
    """Returns total USDT value of the portfolio (USDT + BTC value)"""
    bal = exchange.fetch_balance()
    usdt = bal[QUOTE]['free']
    btc = bal[COIN]['total']
    
    # Get current price to value the BTC
    ticker = exchange.fetch_ticker(SYMBOL)
    price = ticker['last']
    
    total_value = usdt + (btc * price)
    return total_value, price, bal

# --- MAIN LOOP ---
print("Execution Engine Started...")

while True:
    wait_for_next_minute()
    
    # 1. Fetch Shared Data
    data = fetch_live_data(SYMBOL, TIMEFRAME, limit=150)
    if data.empty: continue

    # 2. Get Portfolio State
    try:
        total_portfolio_value, current_price, full_balance = get_portfolio_value()
    except Exception as e:
        print(f"Error fetching balance/ticker: {e}")
        continue

    # 3. Iterate Strategies
    for strat_conf in strategy_configs:
        strategy = strat_conf['strategy']
        name = strat_conf['name']
        alloc_pct = strat_conf['allocation_pct']
        
        # A. Check State Manager for TRUTH
        # We don't trust local variables; we trust the JSON file
        state = state_manager.get_strategy_state(name)
        
        # 'amount' in state is how much BTC this strategy currently owns
        current_btc_holdings = state.get('amount', 0.0)
        
        # Calculate Current Weight (How much of the allocated capital is currently in BTC?)
        # Allocated Capital = Total Portfolio * 0.20
        allocated_capital = total_portfolio_value * alloc_pct
        current_value_held = current_btc_holdings * current_price
        current_weight = current_value_held / allocated_capital if allocated_capital > 0 else 0
        
        print(f"[{name}] Alloc: ${allocated_capital:.2f} | Held: ${current_value_held:.2f} ({current_weight:.2%} invested)")

        # B. Get Signal (Target Weight: 0.0 to 1.0)
        # 1.0 = Fully Invested, 0.0 = Cash, 0.5 = Half Invested
        target_weight = strategy.on_data(data)
        
        # C. Calculate Difference
        # Do we need to buy or sell to reach the target?
        weight_diff = target_weight - current_weight
        
        # Threshold to avoid tiny dust trades (e.g. 1% drift)
        if abs(weight_diff) < 0.02: 
            print(f"[{name}] Signal {target_weight:.2f} close to current {current_weight:.2f}. Holding.")
            continue

        # D. Execute Trade
        try:
            # BUY LOGIC
            if weight_diff > 0:
                amount_usdt_to_spend = weight_diff * allocated_capital
                amount_btc_to_buy = amount_usdt_to_spend / current_price
                
                # Check if we actually have enough USDT in the main wallet
                if full_balance[QUOTE]['free'] < amount_usdt_to_spend:
                    print(f"[{name}] Insufficient USDT funds to execute buy.")
                    continue

                # Precision formatting is REQUIRED for live trading
                final_amount = exchange.amount_to_precision(SYMBOL, amount_btc_to_buy)
                
                print(f"[{name}] BUYING {final_amount} BTC...")
                # order = exchange.create_market_buy_order(SYMBOL, final_amount) # Uncomment to go live
                
                # Update State
                new_total_btc = current_btc_holdings + float(final_amount)
                state_manager.update_position(name, "IN_POSITION", current_price, new_total_btc)

            # SELL LOGIC
            elif weight_diff < 0:
                amount_btc_to_sell = abs(weight_diff * allocated_capital) / current_price
                
                # Cap the sell amount to what we actually own (safety check)
                if amount_btc_to_sell > current_btc_holdings:
                    amount_btc_to_sell = current_btc_holdings
                
                final_amount = exchange.amount_to_precision(SYMBOL, amount_btc_to_sell)
                
                if float(final_amount) == 0: continue

                print(f"[{name}] SELLING {final_amount} BTC...")
                # order = exchange.create_market_sell_order(SYMBOL, final_amount) # Uncomment to go live
                
                # Update State
                new_total_btc = max(0, current_btc_holdings - float(final_amount))
                status = "IN_POSITION" if new_total_btc > 0 else "FLAT"
                state_manager.update_position(name, status, current_price, new_total_btc)

        except Exception as e:
            print(f"[{name}] Trade Failed: {e}")