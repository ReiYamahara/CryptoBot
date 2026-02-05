from mean_reversion import *
from backtesting_engine import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(results, engine, data):
    results = results.sort_values('time').copy()
    results['time'] = pd.to_datetime(results['time'])
    results['ret'] = results['value'].pct_change().fillna(0.0)

    start_value = results.iloc[0]['value']
    final_value = results.iloc[-1]['value']
    net_profit = final_value - start_value
    total_return = (final_value / start_value) - 1 if start_value > 0 else np.nan

    elapsed_sec = (results['time'].iloc[-1] - results['time'].iloc[0]).total_seconds()
    years = elapsed_sec / (365 * 24 * 3600) if elapsed_sec > 0 else np.nan
    cagr = ((final_value / start_value) ** (1 / years) - 1) if years and years > 0 else np.nan

    running_peak = results['value'].cummax()
    drawdown_abs = results['value'] - running_peak
    drawdown_pct = drawdown_abs / running_peak.replace(0, np.nan)
    max_drawdown_abs = drawdown_abs.min()
    max_drawdown_pct = drawdown_pct.min()
    avg_drawdown = drawdown_pct[drawdown_pct < 0].mean()

    max_dd_duration_sec = 0.0
    dd_start = None
    for i in range(len(results)):
        is_underwater = drawdown_pct.iloc[i] < 0
        t = results['time'].iloc[i]
        if is_underwater and dd_start is None:
            dd_start = t
        if (not is_underwater) and dd_start is not None:
            max_dd_duration_sec = max(max_dd_duration_sec, (t - dd_start).total_seconds())
            dd_start = None
    if dd_start is not None:
        end_t = results['time'].iloc[-1]
        max_dd_duration_sec = max(max_dd_duration_sec, (end_t - dd_start).total_seconds())

    bar_sec = results['time'].diff().dt.total_seconds().median()
    periods_per_year = (365 * 24 * 3600 / bar_sec) if pd.notna(bar_sec) and bar_sec > 0 else np.nan
    vol_annual = results['ret'].std(ddof=0) * np.sqrt(periods_per_year) if pd.notna(periods_per_year) else np.nan
    mean_annual = results['ret'].mean() * periods_per_year if pd.notna(periods_per_year) else np.nan
    sharpe = mean_annual / vol_annual if pd.notna(vol_annual) and vol_annual > 0 else np.nan
    var_95 = -results['ret'].quantile(0.05)

    realized = np.array(engine.realized_pnls, dtype=float)
    wins = realized[realized > 0]
    losses = realized[realized < 0]
    win_rate = len(wins) / len(realized) if len(realized) else np.nan
    loss_rate = len(losses) / len(realized) if len(realized) else np.nan
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    win_loss_ratio = (avg_win / abs(avg_loss)) if len(wins) and len(losses) else np.nan

    avg_holding_days = (np.mean(engine.holding_periods_sec) / 86400) if engine.holding_periods_sec else np.nan
    time_in_market = float((results['shares'] > 0).mean())

    vol_data = data.copy()
    if 'time' not in vol_data.columns:
        vol_data['time'] = pd.to_datetime(vol_data['timestamp'], unit='s')
    total_volume = float(vol_data['volume'].sum())
    monthly_volume = vol_data.set_index('time')['volume'].resample('ME').sum()
    avg_monthly_volume = float(monthly_volume.mean()) if not monthly_volume.empty else np.nan

    total_fees = float(engine.total_fees)
    fees_pct_of_return = (total_fees / abs(net_profit)) if net_profit != 0 else np.nan

    return {
        'start_value': start_value,
        'final_value': final_value,
        'net_profit': net_profit,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown_abs': max_drawdown_abs,
        'max_drawdown_pct': max_drawdown_pct,
        'max_dd_duration_days': max_dd_duration_sec / 86400,
        'avg_drawdown': avg_drawdown,
        'vol_annual': vol_annual,
        'var_95': var_95,
        'sharpe': sharpe,
        'num_trades': len(engine.trade_log),
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'avg_holding_days': avg_holding_days,
        'time_in_market': time_in_market,
        'total_volume': total_volume,
        'avg_monthly_volume': avg_monthly_volume,
        'total_fees': total_fees,
        'fees_pct_of_return': fees_pct_of_return,
    }

# 1. LOAD AND PREPARE DATA
csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('btc_data/XBTUSDT_5.csv', header=None, names=csv_headers)
# df['time'] = pd.to_datetime(df['timestamp'], unit='s')

def run_backtest(fee, label):
    strategy = MeanReversionStrategy(window=50, z_threshold=3)
    engine = BackTestEngine(initial_capital=10000.0, fee=fee)
    print(f"Starting Backtest ({label})...")
    engine.run(df, strategy)
    print(f"Backtest Complete ({label}).")
    results = pd.DataFrame(engine.equity_curve)
    metrics = compute_metrics(results, engine, df)
    return results, metrics

# 2. RUN WITH AND WITHOUT FEES
results_fee, metrics_fee = run_backtest(0.001, "with fees")
results_no_fee, metrics_no_fee = run_backtest(0.0, "no fees")

def print_metrics(metrics, label):
    print(f"\n=== Results ({label}) ===")
    print(f"Initial Portfolio: ${metrics['start_value']:,.2f}")
    print(f"Final Portfolio:   ${metrics['final_value']:,.2f}")
    print(f"Net Profit:        ${metrics['net_profit']:,.2f}")
    print(f"Total Return:      {metrics['total_return'] * 100:.2f}%")
    print(f"CAGR:              {metrics['cagr'] * 100:.2f}%")
    print(f"Max Drawdown:      ${metrics['max_drawdown_abs']:,.2f} ({metrics['max_drawdown_pct'] * 100:.2f}%)")
    print(f"Max DD Duration:   {metrics['max_dd_duration_days']:.2f} days")
    print(f"Average Drawdown:  {metrics['avg_drawdown'] * 100:.2f}%")
    print(f"Volatility (Ann):  {metrics['vol_annual'] * 100:.2f}%")
    print(f"VaR 95% (1-bar):   {metrics['var_95'] * 100:.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe']:.2f}")
    print(f"Number of Trades:  {metrics['num_trades']}")
    print(f"Win Rate:          {metrics['win_rate'] * 100:.2f}%")
    print(f"Loss Rate:         {metrics['loss_rate'] * 100:.2f}%")
    print(f"Average Win:       ${metrics['avg_win']:,.2f}")
    print(f"Average Loss:      ${metrics['avg_loss']:,.2f}")
    print(f"Win/Loss Ratio:    {metrics['win_loss_ratio']:.2f}")
    print(f"Avg Hold Period:   {metrics['avg_holding_days']:.2f} days")
    print(f"Time in Market:    {metrics['time_in_market'] * 100:.2f}%")
    print(f"Total Volume:      {metrics['total_volume']:,.2f}")
    print(f"Avg Monthly Vol:   {metrics['avg_monthly_volume']:,.2f}")
    print(f"Total Fees Paid:   ${metrics['total_fees']:,.2f}")
    print(f"Fees % of Return:  {metrics['fees_pct_of_return'] * 100:.2f}%")

print_metrics(metrics_fee, "with fees")
print_metrics(metrics_no_fee, "no fees")

# 5. VISUALIZE
plt.figure(figsize=(10, 6))
plt.plot(results_fee['time'], results_fee['value'], label='Equity (with fees)')
plt.plot(results_no_fee['time'], results_no_fee['value'], label='Equity (no fees)')
plt.title('Backtest Results (With vs Without Fees)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()
