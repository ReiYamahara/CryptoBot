import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from strategy.RSI import *
from backtest.backtesting_engine import *
from backtest.compute_metrics import compute_metrics

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# CONFIG
PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
ENTRY_CONFIRM_BARS = 2
INITIAL_CAPITAL = 10000.0
FEE_ON = 0.001
START_DATE = None  # e.g., "2024-01-01"
END_DATE = None    # e.g., "2024-12-31"
VEL_EMA_N = 4

# 1. LOAD AND PREPARE DATA
csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('btc_data/XBTUSDT_5.csv', header=None, names=csv_headers)
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
if START_DATE:
    df = df[df['time'] >= pd.to_datetime(START_DATE)]
if END_DATE:
    df = df[df['time'] <= pd.to_datetime(END_DATE)]
if not START_DATE and not END_DATE:
    max_time = df['time'].max()
    cutoff_time = max_time - pd.DateOffset(years=2)
    df = df[df['time'] >= cutoff_time]
df = df.reset_index(drop=True)

def run_backtest(fee, label):
    strategy = RSIStrategy(
        period=PERIOD,
        oversold=OVERSOLD,
        overbought=OVERBOUGHT,
        entry_confirm_bars=ENTRY_CONFIRM_BARS,
    )
    engine = BackTestEngine(initial_capital=INITIAL_CAPITAL, fee=fee)
    print(f"Starting Backtest ({label})...")
    engine.run(df, strategy)
    print(f"Backtest Complete ({label}).")
    results = pd.DataFrame(engine.equity_curve)
    metrics = compute_metrics(results, engine, df, strategy=strategy)
    return results, metrics, engine

# 2. RUN WITH AND WITHOUT FEES
results_fee, metrics_fee, engine_fee = run_backtest(FEE_ON, "with fees")
results_no_fee, metrics_no_fee, engine_no_fee = run_backtest(0.0, "no fees")

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
        ("Stop Loss Hits", "stop_loss_hits", "int"),
    ]

    header = f"{'Metric':<22} | {'With Fees':>15} | {'No Fees':>15}"
    print("\n" + header)
    print("-" * len(header))
    for label, key, kind in rows:
        fee_val = _fmt(metrics_fee.get(key), kind)
        no_fee_val = _fmt(metrics_no_fee.get(key), kind)
        print(f"{label:<22} | {fee_val:>15} | {no_fee_val:>15}")

print_side_by_side(metrics_fee, metrics_no_fee)

# 5. SIGNAL DATA (RSI)
signal_strategy = RSIStrategy(
    period=PERIOD,
    oversold=OVERSOLD,
    overbought=OVERBOUGHT,
    entry_confirm_bars=ENTRY_CONFIRM_BARS,
)
signals = []
rsi_vals = []
entry_streaks = []
for _, row in df.iterrows():
    signals.append(signal_strategy.on_data(row))
    rsi_vals.append(signal_strategy.last_rsi)
    entry_streaks.append(signal_strategy.last_entry_streak)

df['signal'] = signals
df['signal_change'] = df['signal'].diff().fillna(0.0)
df['rsi'] = rsi_vals
df['entry_streak'] = entry_streaks
buys = df[df['signal_change'] > 0]
sells = df[df['signal_change'] < 0]

# Velocity / deceleration for visibility
velocities = []
d_velocities = []
prev_log_p = None
velocity_ema = None
prev_velocity_ema = None
vel_alpha = 2 / (VEL_EMA_N + 1)
for price in df['close']:
    log_p = np.log(price)
    if prev_log_p is None:
        prev_log_p = log_p
        velocities.append(np.nan)
        d_velocities.append(np.nan)
        continue
    log_ret = log_p - prev_log_p
    prev_log_p = log_p
    if velocity_ema is None:
        velocity_ema = log_ret
    else:
        velocity_ema = (vel_alpha * log_ret) + ((1 - vel_alpha) * velocity_ema)
    d_vel = np.nan if prev_velocity_ema is None else (velocity_ema - prev_velocity_ema)
    prev_velocity_ema = velocity_ema
    velocities.append(velocity_ema)
    d_velocities.append(d_vel)

df['velocity_ema'] = velocities
df['d_velocity'] = d_velocities

def _trades_per_6h(engine):
    if not engine.trade_log:
        return pd.Series(dtype=float)
    trade_df = pd.DataFrame(engine.trade_log)
    trade_df['time'] = pd.to_datetime(trade_df['time'])
    return trade_df.set_index('time').resample('6h').size()

trades_fee = _trades_per_6h(engine_fee)
time_in_market_6h = df.set_index('time')['signal'].gt(0).resample('6h').mean() * 100.0

# 6. COMBINED PLOT (EQUITY + SIGNALS)
os.makedirs("backtest/plots", exist_ok=True)
fig, (ax_eq, ax_price, ax_rsi, ax_vel, ax_trades) = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# Equity curve
ax_eq.plot(results_fee['time'], results_fee['value'], label='Equity (with fees)')
ax_eq.plot(results_no_fee['time'], results_no_fee['value'], label='Equity (no fees)')
ax_eq.set_title('RSI Backtest Results (With vs Without Fees)')
ax_eq.set_ylabel('Portfolio Value ($)')
ax_eq.legend()
ax_eq.grid(True)

# Price + signals
ax_price.plot(df['time'], df['close'], label='Price', color='black')
ax_price.scatter(buys['time'], buys['close'], marker='^', color='green', label='Buy', s=40)
ax_price.scatter(sells['time'], sells['close'], marker='v', color='red', label='Sell', s=40)
ax_price.set_ylabel('Price')
ax_price.legend()
ax_price.grid(True)

# RSI panel
ax_rsi.plot(df['time'], df['rsi'], label='RSI', color='blue')
ax_rsi.axhline(OVERSOLD, color='red', linestyle='--', label='Oversold')
ax_rsi.axhline(OVERBOUGHT, color='gray', linestyle='--', label='Overbought')
ax_rsi.set_ylabel('RSI')
ax_rsi.legend()
ax_rsi.grid(True)

# Velocity + deceleration
ax_vel.plot(df['time'], df['velocity_ema'], label='Velocity (EMA log returns)', color='tab:blue')
ax_vel.plot(df['time'], df['d_velocity'], label='Deceleration (Δ velocity)', color='tab:orange')
ax_vel.axhline(0, color='gray', linestyle='--', linewidth=1)
ax_vel.set_ylabel('Velocity')
ax_vel.legend()
ax_vel.grid(True)

# Trade frequency + time in market (dual axis)
if not trades_fee.empty:
    ax_trades.plot(trades_fee.index, trades_fee.values, label='Trades/6h (with fees)', color='tab:orange')
ax_trades.set_ylabel('Trades/6h')
ax_trades.grid(True)

ax_tim = ax_trades.twinx()
ax_tim.plot(time_in_market_6h.index, time_in_market_6h.values, label='Time in Market (%)', color='tab:purple', alpha=0.7)
ax_tim.set_ylabel('Time in Market (%)')

lines1, labels1 = ax_trades.get_legend_handles_labels()
lines2, labels2 = ax_tim.get_legend_handles_labels()
ax_trades.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("backtest/plots/rsi_plots.png", dpi=150, bbox_inches="tight")
plt.close()
