import pandas as pd

class BackTestEngine:
    def __init__(self, initial_capital=1000.00, fee=0.00, print_trades=False):
        self.cash = initial_capital
        self.fee = fee # 0.0 disables fees; non-zero uses fixed 0.35% taker fee
        self.print_trades = print_trades
        self.shares = 0.0
        self.portfolio_value = initial_capital
        self.equity_curve = []
        self.trade_log = []
        self.realized_pnls = []
        self.avg_entry_price = 0.0
        self.position_open_time = None
        self.holding_periods_sec = []
        self.total_fees = 0.0
        self.total_traded_notional = 0.0
        self.last_target_pct = None
    def _current_taker_fee(self):
        return 0.0 if self.fee == 0 else 0.0035

    def run(self, data, strategy):
        if not 'time' in data.columns:
            data['time'] = pd.to_datetime(data['timestamp'], unit='s')
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['close']
            target_pct = strategy.on_data(row)
            if self.last_target_pct is None or target_pct != self.last_target_pct:
                self.rebalance(target_pct, price, row['time'])
                self.last_target_pct = target_pct

            total_value = self.cash + (self.shares*price)
            self.equity_curve.append({
                'time': row['time'],
                'value': total_value,
                'shares': self.shares
            })

    def rebalance(self, target_pct, price, time):
        cur_total_val = self.cash + (self.shares * price)
        target_holding_val = cur_total_val * target_pct
        
        cur_holding_val = self.shares * price
        diff_value = target_holding_val - cur_holding_val

        # buy more
        if diff_value > 0:
            fee_rate = self._current_taker_fee()
            available_for_trade = min(diff_value, self.cash / (1 + fee_rate))
            qty_to_buy = available_for_trade / price
            if qty_to_buy > 0:
                notional = qty_to_buy * price
                fee_cost = notional * fee_rate
                self.total_fees += fee_cost
                self.total_traded_notional += notional
                prev_shares = self.shares
                prev_cost = prev_shares * self.avg_entry_price
                self.shares += qty_to_buy
                self.cash -= notional + fee_cost
                total_cost = prev_cost + notional + fee_cost
                self.avg_entry_price = total_cost / self.shares if self.shares > 0 else 0.0
                if prev_shares == 0:
                    self.position_open_time = time
                self.trade_log.append({'time': time, 'side': 'BUY', 'notional': notional})
                if self.print_trades:
                    print(f"[{time}] BUY: {qty_to_buy:.4f} shares @ ${price:.2f}")

        else: # sell off
            qty_to_sell = min(abs(diff_value) / price, self.shares)
            if qty_to_sell > 0:
                notional_received = qty_to_sell * price
                fee_rate = self._current_taker_fee()
                fee_cost = notional_received * fee_rate
                self.total_fees += fee_cost
                self.total_traded_notional += notional_received
                realized_pnl = (notional_received - fee_cost) - (qty_to_sell * self.avg_entry_price)
                self.shares -= qty_to_sell
                self.cash += notional_received - fee_cost
                self.realized_pnls.append(realized_pnl)
                self.trade_log.append({'time': time, 'side': 'SELL', 'realized_pnl': realized_pnl, 'notional': notional_received})
                if self.shares == 0:
                    if self.position_open_time is not None:
                        hold_seconds = (time - self.position_open_time).total_seconds()
                        self.holding_periods_sec.append(hold_seconds)
                        self.position_open_time = None
                    self.avg_entry_price = 0.0
                if self.print_trades:
                    print(f"[{time}] SELL: {qty_to_sell:.4f} shares @ ${price:.2f}")
        
