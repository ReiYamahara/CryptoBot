import numpy as np
from collections import deque
from strategy.strategy_base import StrategyBase


class RSIStrategy(StrategyBase):
    def __init__(self, period=14, oversold=30, overbought=70, entry_confirm_bars=2):
        super().__init__("RSI")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.entry_confirm_bars = max(1, int(entry_confirm_bars))
        self.prices = deque(maxlen=period + 1)
        self.position = 0.0
        self.stop_loss_events = []
        self._entry_streak = 0
        self.last_rsi = np.nan
        self.last_entry_streak = 0

    def _rsi(self):
        diffs = np.diff(np.array(self.prices))
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_data(self, row):
        current_price = row['close']
        self.prices.append(current_price)

        if len(self.prices) < self.period + 1:
            self.last_rsi = np.nan
            self.last_entry_streak = self._entry_streak
            return 0.0

        rsi = self._rsi()
        self.last_rsi = rsi

        if rsi <= self.oversold:
            self._entry_streak += 1
        else:
            self._entry_streak = 0

        if self.position == 0.0 and self._entry_streak >= self.entry_confirm_bars:
            self.position = 1.0
        elif rsi >= self.overbought:
            self.position = 0.0
            self._entry_streak = 0

        self.last_entry_streak = self._entry_streak

        return self.position
