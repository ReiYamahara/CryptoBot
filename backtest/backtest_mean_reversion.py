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
from collections import deque

# CONFIG
WINDOW = 200
ENTRY_Z = -2
EXIT_Z = 2
STOP_LOSS_PCT = 0.02 # 3%
POSITION_SIZE = 1.0
INITIAL_CAPITAL = 10000.0
FEE_ON = 0.0035 # 0.35%
START_DATE = "2025-08-01"  # e.g., "2024-01-01"
END_DATE = "2025-10-01"    # e.g., "2024-12-31"
VEL_EMA_N = 5
VEL_STD_WINDOW = 50
VEL_K = 0.7
VEL_CAP = 2.0
ENTRY_CONFIRM_BARS = 2

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
    strategy = MeanReversionStrategy(
        window=WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        stop_loss_pct=STOP_LOSS_PCT,
        position_size=POSITION_SIZE,
        vel_ema_n=VEL_EMA_N,
        vel_std_window=VEL_STD_WINDOW,
        vel_k=VEL_K,
        vel_cap=VEL_CAP,
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

# 5. SIGNAL DATA (MEAN REVERSION)
signal_strategy = MeanReversionStrategy(
    window=WINDOW,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    stop_loss_pct=STOP_LOSS_PCT,
    position_size=POSITION_SIZE,
    vel_ema_n=VEL_EMA_N,
    vel_std_window=VEL_STD_WINDOW,
    vel_k=VEL_K,
    vel_cap=VEL_CAP,
    entry_confirm_bars=ENTRY_CONFIRM_BARS,
)
signals = []
for _, row in df.iterrows():
    signals.append(signal_strategy.on_data(row))

df['signal'] = signals
df['signal_change'] = df['signal'].diff().fillna(0.0)
buys = df[df['signal_change'] > 0]
sells = df[df['signal_change'] < 0]
stop_times = [t for t, _ in signal_strategy.stop_loss_events]
stop_prices = [p for _, p in signal_strategy.stop_loss_events]

# Z-score series using log price, EMA(log price), and STD(log price)
log_hist = deque(maxlen=WINDOW)
log_return_hist = deque(maxlen=VEL_STD_WINDOW)
z_scores = []
entry_z_dynamic = []
alpha = 2 / (WINDOW + 1)
vel_alpha = 2 / (VEL_EMA_N + 1)
prev_log_p = None
velocity_ema = None
for price in df['close']:
    log_p = np.log(price)
    if prev_log_p is not None:
        log_ret = log_p - prev_log_p
        log_return_hist.append(log_ret)
        if velocity_ema is None:
            velocity_ema = log_ret
        else:
            velocity_ema = (vel_alpha * log_ret) + ((1 - vel_alpha) * velocity_ema)
    prev_log_p = log_p

    log_hist.append(log_p)
    if len(log_hist) < WINDOW:
        z_scores.append(np.nan)
        entry_z_dynamic.append(ENTRY_Z)
        continue
    history = np.array(log_hist)
    ema = history[0]
    for v in history[1:]:
        ema = (alpha * v) + ((1 - alpha) * ema)
    std = np.std(history)
    z_scores.append((log_p - ema) / std if std != 0 else 0.0)

    velocity_z = 0.0
    if len(log_return_hist) >= VEL_STD_WINDOW and velocity_ema is not None:
        vel_std = np.std(np.array(log_return_hist))
        vel_std = max(vel_std, 1e-8)
        velocity_z = velocity_ema / vel_std
    downside_velocity = max(0.0, -velocity_z)
    downside_velocity = min(downside_velocity, VEL_CAP)
    entry_z_dynamic.append(ENTRY_Z - (VEL_K * downside_velocity))

df['z_score'] = z_scores
df['entry_z_dynamic'] = entry_z_dynamic

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

def _trades_per_day(engine):
    if not engine.trade_log:
        return pd.Series(dtype=float)
    trade_df = pd.DataFrame(engine.trade_log)
    trade_df['time'] = pd.to_datetime(trade_df['time'])
    return trade_df.set_index('time').resample('6h').size()

trades_fee = _trades_per_day(engine_fee)
time_in_market_6h = df.set_index('time')['signal'].gt(0).resample('6h').mean() * 100.0

# 6. COMBINED PLOT (EQUITY + SIGNALS)
os.makedirs("backtest/plots", exist_ok=True)
fig, (ax_eq, ax_price, ax_z, ax_vel, ax_trades) = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# Equity curve
ax_eq.plot(results_fee['time'], results_fee['value'], label='Equity (with fees)')
ax_eq.plot(results_no_fee['time'], results_no_fee['value'], label='Equity (no fees)')
ax_eq.set_title('Backtest Results (With vs Without Fees)')
ax_eq.set_ylabel('Portfolio Value ($)')
ax_eq.legend()
ax_eq.grid(True)

# Price + signals
ax_price.plot(df['time'], df['close'], label='Price', color='black')
ax_price.scatter(buys['time'], buys['close'], marker='^', color='green', label='Buy', s=40)
ax_price.scatter(sells['time'], sells['close'], marker='v', color='red', label='Sell', s=40)
if stop_times:
    ax_price.scatter(stop_times, stop_prices, marker='x', color='orange', label='Stop Loss', s=40)
ax_price.set_ylabel('Price')
ax_price.legend()
ax_price.grid(True)

# Z-score panel
ax_z.plot(df['time'], df['z_score'], label='Z-score', color='blue')
ax_z.axhline(ENTRY_Z, color='red', linestyle='--', label='Entry (base)')
ax_z.plot(df['time'], df['entry_z_dynamic'], color='purple', linestyle=':', label='Entry (dynamic)')
ax_z.axhline(EXIT_Z, color='gray', linestyle='--', label='Exit (+z)')
if stop_times:
    for i, t in enumerate(stop_times):
        ax_z.axvline(t, color='orange', linestyle=':', alpha=0.4, label='Stop-loss breach' if i == 0 else None)
ax_z.set_ylabel('Z-score')
ax_z.legend()
ax_z.grid(True)

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
plt.savefig("backtest/plots/mean_reversion_plots.png", dpi=150, bbox_inches="tight")
plt.close()
