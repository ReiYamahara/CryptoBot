from pull_data import *

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


import numpy as np
from collections import deque
from strategy_base import StrategyBase

class MeanReversionStrategy(StrategyBase):
    def __init__(self, window=50, z_threshold=2.0):
        super().__init__("MeanReversion_LiveReady")
        self.window = window
        self.z_threshold = z_threshold
        
        self.price_history = deque(maxlen=window) 
        
        self.position = 0.0 # 0.0 = Neutral, 1.0 = Long

    def on_data(self, row):
        """
        Input: row['close'] (The current price only)
        """
        current_price = row['close']
        
        # 1. Update Memory
        self.price_history.append(current_price)
        
        # 2. Check if we have enough data to make a decision
        if len(self.price_history) < self.window:
            return 0.0
            
        # converting queue to array for faster math
        history = np.array(self.price_history)
        
        sma = np.mean(history)
        std = np.std(history)
        
        # Avoid division by zero
        if std == 0:
            return 0.0
            
        z_score = (current_price - sma) / std
        
        # 4. Trading Logic
        if z_score < -self.z_threshold:
            self.position = 0.3 # Buy
        elif z_score >= 0:
            self.position = 0.0 # Sell
            
        return self.position


