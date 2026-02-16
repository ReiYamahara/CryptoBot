# Strategies

This folder contains trading strategies that implement the common `StrategyBase` interface.

## Files
- `strategy_base.py` — Base class for strategies.
- `mean_reversion.py` — Z-score mean reversion strategy.
- `RSI.py` — RSI strategy (oversold/overbought).

## Notes
Strategies return a **target position percentage** (0.0–1.0). The backtest engine handles execution and sizing.
