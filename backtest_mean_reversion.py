from mean_reversion import *
from backtesting_engine import *

import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD AND PREPARE DATA
csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('master_q4/XBTUSDT_5.csv', header=None, names=csv_headers)
# df['time'] = pd.to_datetime(df['timestamp'], unit='s')

# 2. INITIALIZE
# Create the strategy instance
my_strategy = MeanReversionStrategy(window=50, z_threshold=3)

# Create the engine instance
engine = BackTestEngine(initial_capital=10000.0, fee=0.001)

# 3. RUN
print("Starting Backtest...")
engine.run(df, my_strategy)
print("Backtest Complete.")

# 4. ANALYZE RESULTS
# Convert the result list to a DataFrame for easy analysis
results = pd.DataFrame(engine.equity_curve)

# Print final stats
final_value = results.iloc[-1]['value']
start_value = results.iloc[0]['value']
total_return = ((final_value - start_value) / start_value) * 100

print(f"Initial Portfolio: ${start_value:,.2f}")
print(f"Final Portfolio:   ${final_value:,.2f}")
print(f"Total Return:      {total_return:.2f}%")

# 5. VISUALIZE
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['value'], label='Equity Curve')
plt.title('Backtest Results')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()