from pull_data import *

# SIMPLE MEAN REVERSION CODE THAT JUST RUNS ON ITS OWN (DOES NOT WORK WITH THE LIVE TRADING PLATFORM)
df['sma'] = df['close'].rolling(window=50).mean()
df['std'] = df['close'].rolling(window=50).std()
df['time'] = pd.to_datetime(df['timestamp'], unit='s')

df['z_score'] = (df['close'] - df['sma'])/df['std']

# Initialize variables
cash = 1000.00
shares = 0.0
allocation_pct = 0.3
epsilon = 1e-1

for i in range(len(df)):

    current_price = df['close'].iloc[i]
    z = df['z_score'].iloc[i]
    
    # --- BUY ACTION ---
    if z < -2 and shares == 0:
        invest_amount = cash * allocation_pct
        shares = invest_amount / current_price  
        cash -= invest_amount
        print(f"Time {df['time'].iloc[i]}: Bought {shares:.2f} shares at ${current_price:.2f}")

    # --- SELL ACTION ---
    if z >= 0 and shares > 0:
        revenue = shares * current_price        
        cash += revenue
        shares = 0.0                           
        print(f"Time {df['time'].iloc[i]}: Sold at ${current_price:.2f}. New Cash: ${cash:.2f}")

# Calculate final portfolio value
final_value = cash + (shares * df['close'].iloc[-1])
print(f"Final Portfolio Value: ${final_value:.2f}")
print(cash)