import numpy as np
import pandas as pd


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

    daily_equity = results.set_index('time')['value'].resample('D').last().dropna()
    daily_ret = daily_equity.pct_change().dropna()
    if len(daily_ret) > 1:
        vol_annual = daily_ret.std(ddof=0) * np.sqrt(365)
        mean_annual = daily_ret.mean() * 365
        sharpe = mean_annual / vol_annual if vol_annual > 0 else np.nan
        var_95 = -daily_ret.quantile(0.05)
    else:
        vol_annual = np.nan
        sharpe = np.nan
        var_95 = np.nan

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

    trade_log = pd.DataFrame(engine.trade_log)
    if not trade_log.empty and 'notional' in trade_log.columns:
        trade_log['time'] = pd.to_datetime(trade_log['time'])
        total_traded_notional = float(trade_log['notional'].sum())
        monthly_notional = trade_log.set_index('time')['notional'].resample('ME').sum()
        avg_monthly_traded_notional = float(monthly_notional.mean()) if not monthly_notional.empty else np.nan
    else:
        total_traded_notional = float(getattr(engine, 'total_traded_notional', 0.0))
        avg_monthly_traded_notional = np.nan

    total_fees = float(engine.total_fees)

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
        'num_trades': len(engine.realized_pnls),
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'avg_holding_days': avg_holding_days,
        'time_in_market': time_in_market,
        'total_traded_notional': total_traded_notional,
        'avg_monthly_traded_notional': avg_monthly_traded_notional,
        'total_fees': total_fees,
    }
