import numpy as np
from collections import deque
from strategy_base import StrategyBase

class MeanReversionStrategy(StrategyBase):
    def __init__(self, window=50, z_threshold=2.0, stop_loss_pct=0.05):
        super().__init__("MeanReversion_LiveReady")
        self.window = window
        self.z_threshold = z_threshold
        self.stop_loss_pct = stop_loss_pct
        
        self.price_history = deque(maxlen=window) 
        
        self.position = 0.0 # 0.0 = Neutral, 1.0 = Long
        self.entry_price = 0.0

    def on_data(self, row):
        """
        Input: row['close'] (The current price only)
        """
        latest_candle = row.iloc[-1]
        
        current_price = float(latest_candle['close'])

        if len(self.price_history) == 0:
            self.price_history.append(current_price)
            return 0.0

        # Stop loss logic
        if self.position > 0:
            # If price falls X% below entry, GTFO
            if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                self.position = 0.0
                self.entry_price = 0.0
                self.price_history.append(current_price)
                stop_time = row['time'] if 'time' in row else 'unknown time'
                print(f"STOP LOSS @ {stop_time}")
                return self.position

        # Converting queue to array for faster math
        history = np.array(self.price_history)
        
        sma = np.mean(history)
        std = np.std(history)
            
        z_score = (current_price - sma) / std if std != 0 else 0

        # Update Memory
        self.price_history.append(current_price)

        # Check if we have enough data to make a decision
        if len(self.price_history) < self.window:
            return 0.0

        print(f"Current z-score is: {z_score}")
        # 4. Trading Logic
        if z_score < -self.z_threshold:
            self.position = 1.0 # Buy
            self.entry_price = current_price
        elif z_score >= 0:
            self.position = 0.0 # Sell
            
        return self.position
