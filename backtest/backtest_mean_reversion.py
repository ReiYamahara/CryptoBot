import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from strategy.mean_reversion import *
from backtest.backtesting_engine import *
from backtest.compute_metrics import compute_metrics

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG
WINDOW = 100
Z_THRESHOLD = 3
STOP_LOSS_PCT = 0.02 # 2%
POSITION_SIZE = 1.0
INITIAL_CAPITAL = 10000.0
FEE_ON = 0.0035 # 0.35%

# 1. LOAD AND PREPARE DATA
csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('btc_data/XBTUSDT_5.csv', header=None, names=csv_headers)
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
max_time = df['time'].max()
cutoff_time = max_time - pd.DateOffset(years=2)
df = df[df['time'] >= cutoff_time].reset_index(drop=True)

def run_backtest(fee, label):
    strategy = MeanReversionStrategy(
        window=WINDOW,
        z_threshold=Z_THRESHOLD,
        stop_loss_pct=STOP_LOSS_PCT,
        position_size=POSITION_SIZE,
    )
    engine = BackTestEngine(initial_capital=INITIAL_CAPITAL, fee=fee)
    print(f"Starting Backtest ({label})...")
    engine.run(df, strategy)
    print(f"Backtest Complete ({label}).")
    results = pd.DataFrame(engine.equity_curve)
    metrics = compute_metrics(results, engine, df)
    return results, metrics

# 2. RUN WITH AND WITHOUT FEES
results_fee, metrics_fee = run_backtest(FEE_ON, "with fees")
results_no_fee, metrics_no_fee = run_backtest(0.0, "no fees")

def _fmt(val, kind):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    if kind == "money":
        return f"${val:,.2f}"
    if kind == "pct":
        return f"{val * 100:.2f}%"
    if kind == "days":
        return f"{val:.2f} days"
    if kind == "int":
        return f"{int(val)}"
    return f"{val:.2f}"

def print_side_by_side(metrics_fee, metrics_no_fee):
    rows = [
        ("Initial Portfolio", "start_value", "money"),
        ("Final Portfolio", "final_value", "money"),
        ("Net Profit", "net_profit", "money"),
        ("Total Return", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Max Drawdown ($)", "max_drawdown_abs", "money"),
        ("Max Drawdown (%)", "max_drawdown_pct", "pct"),
        ("Max DD Duration", "max_dd_duration_days", "days"),
        ("Average Drawdown", "avg_drawdown", "pct"),
        ("Volatility (Ann)", "vol_annual", "pct"),
        ("VaR 95% (Daily)", "var_95", "pct"),
        ("Sharpe Ratio", "sharpe", "num"),
        ("Round-Trip Trades", "num_trades", "int"),
        ("Win Rate", "win_rate", "pct"),
        ("Loss Rate", "loss_rate", "pct"),
        ("Average Win", "avg_win", "money"),
        ("Average Loss", "avg_loss", "money"),
        ("Win/Loss Ratio", "win_loss_ratio", "num"),
        ("Avg Hold Period", "avg_holding_days", "days"),
        ("Time in Market", "time_in_market", "pct"),
        ("Total Traded", "total_traded_notional", "money"),
        ("Avg Monthly Trd", "avg_monthly_traded_notional", "money"),
        ("Total Fees Paid", "total_fees", "money"),
    ]

    header = f"{'Metric':<22} | {'With Fees':>15} | {'No Fees':>15}"
    print("\n" + header)
    print("-" * len(header))
    for label, key, kind in rows:
        fee_val = _fmt(metrics_fee.get(key), kind)
        no_fee_val = _fmt(metrics_no_fee.get(key), kind)
        print(f"{label:<22} | {fee_val:>15} | {no_fee_val:>15}")

print_side_by_side(metrics_fee, metrics_no_fee)

# 5. SIGNAL PLOT (MEAN REVERSION)
signal_strategy = MeanReversionStrategy(window=50, z_threshold=3)
signals = []
for _, row in df.iterrows():
    signals.append(signal_strategy.on_data(row))

df['signal'] = signals
df['signal_change'] = df['signal'].diff().fillna(0.0)
buys = df[df['signal_change'] > 0]
sells = df[df['signal_change'] < 0]
stop_times = [t for t, _ in signal_strategy.stop_loss_events]
stop_prices = [p for _, p in signal_strategy.stop_loss_events]

os.makedirs("backtest/plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(df['time'], df['close'], label='Price', color='black')
ax1.scatter(buys['time'], buys['close'], marker='^', color='green', label='Buy', s=40)
ax1.scatter(sells['time'], sells['close'], marker='v', color='red', label='Sell', s=40)
if stop_times:
    ax1.scatter(stop_times, stop_prices, marker='x', color='orange', label='Stop Loss', s=40)
ax1.legend()
ax1.grid(True)

ax2.step(df['time'], df['signal'], where='post', label='Signal', color='blue')
ax2.set_ylim(-0.05, 1.05)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("backtest/plots/mean_reversion_signals.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. VISUALIZE
os.makedirs("backtest/plots", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(results_fee['time'], results_fee['value'], label='Equity (with fees)')
plt.plot(results_no_fee['time'], results_no_fee['value'], label='Equity (no fees)')
plt.title('Backtest Results (With vs Without Fees)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.savefig("backtest/plots/mean_reversion_equity.png", dpi=150, bbox_inches="tight")
plt.close()
