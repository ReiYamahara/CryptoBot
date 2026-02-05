# Backtests

This folder contains the backtesting engine and runnable backtest scripts.

## Files
- `backtesting_engine.py` — Core backtest engine (rebalances positions, tracks equity, fees, and trades).
- `compute_metrics.py` — Shared performance metrics for backtests.
- `backtest_mean_reversion.py` — Mean reversion backtest runner.
- `backtest_RSI.py` — RSI backtest runner.

## Usage
From the repo root:

```bash
python3 backtest/backtest_mean_reversion.py
python3 backtest/backtest_RSI.py
```

Plots are saved to `backtest/plots/`.
