import numpy as np
from collections import deque
from strategy.strategy_base import StrategyBase

class MeanReversionStrategy(StrategyBase):
    def __init__(
        self,
        window=50,
        entry_z=2.0,
        exit_z=2.0,
        stop_loss_pct=0.05,
        position_size=1.0,
        vel_ema_n=5,
        vel_std_window=30,
        vel_k=1.0,
        vel_cap=3.0,
        entry_confirm_bars=2,
    ):
        super().__init__("MeanReversion_LiveReady")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_pct = stop_loss_pct
        self.position_size = max(0.0, min(1.0, position_size))
        self.vel_ema_n = vel_ema_n
        self.vel_std_window = vel_std_window
        self.vel_k = vel_k
        self.vel_cap = vel_cap
        self.entry_confirm_bars = max(1, int(entry_confirm_bars))
        
        self.log_price_history = deque(maxlen=window) 
        self.log_return_history = deque(maxlen=vel_std_window)
        self.prev_log_price = None
        self.velocity_ema = None
        self.prev_velocity_ema = None
        self._entry_streak = 0
        
        self.position = 0.0 # 0.0 = Neutral, 1.0 = Long
        self.entry_price = 0.0
        self.stop_loss_events = []

    def on_data(self, row):
        """
        Input: row['close'] (The current price only)
        """
<<<<<<< Updated upstream
        latest_candle = row.iloc[-1]
        
        current_price = float(latest_candle['close'])
=======
        current_price = row['close']
        current_log = np.log(current_price)
>>>>>>> Stashed changes

        if self.prev_log_price is None:
            self.prev_log_price = current_log
            self.log_price_history.append(current_log)
            return 0.0

        log_return = current_log - self.prev_log_price
        self.prev_log_price = current_log
        self.log_return_history.append(log_return)
        self.prev_velocity_ema = self.velocity_ema
        if self.velocity_ema is None:
            self.velocity_ema = log_return
        else:
            vel_alpha = 2 / (self.vel_ema_n + 1)
            self.velocity_ema = (vel_alpha * log_return) + ((1 - vel_alpha) * self.velocity_ema)

        # Stop loss logic
        if self.position > 0:
            # If price falls X% below entry, GTFO
            if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                self.position = 0.0
                self.entry_price = 0.0
                self._entry_streak = 0
                self.log_price_history.append(current_log)
                stop_time = row['time'] if 'time' in row else 'unknown time'
                self.stop_loss_events.append((stop_time, current_price))
                return self.position

        # Update memory with log price
        self.log_price_history.append(current_log)

        # Check if we have enough data to make a decision
        if len(self.log_price_history) < self.window:
            return 0.0

<<<<<<< Updated upstream
        print(f"Current z-score is: {z_score}")
=======
        # Converting queue to array for faster math
        history = np.array(self.log_price_history)

        # EMA of log prices
        alpha = 2 / (self.window + 1)
        ema = history[0]
        for v in history[1:]:
            ema = (alpha * v) + ((1 - alpha) * ema)

        std = np.std(history)
        z_score = (current_log - ema) / std if std != 0 else 0

        velocity_z = 0.0
        if len(self.log_return_history) >= self.vel_std_window and self.velocity_ema is not None:
            vel_std = np.std(np.array(self.log_return_history))
            vel_std = max(vel_std, 1e-8)
            velocity_z = self.velocity_ema / vel_std
        downside_velocity = max(0.0, -velocity_z)
        downside_velocity = min(downside_velocity, self.vel_cap)
        entry_z_dynamic = self.entry_z - (self.vel_k * downside_velocity)

>>>>>>> Stashed changes
        # 4. Trading Logic
        if z_score < entry_z_dynamic:
            self._entry_streak += 1
        else:
            self._entry_streak = 0

        decel_ok = (
            self.velocity_ema is not None
            and self.prev_velocity_ema is not None
            and self.velocity_ema < 0
            and (self.velocity_ema - self.prev_velocity_ema) > 0
        )

        if self.position == 0.0 and self._entry_streak >= self.entry_confirm_bars and decel_ok:
            self.position = self.position_size # Buy
            self.entry_price = current_price
        elif z_score > self.exit_z:
            self.position = 0.0 # Sell
            self._entry_streak = 0
            
        return self.position
