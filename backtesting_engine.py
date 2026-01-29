class BackTestEngine:
    def __init__(self, initial_capital=1000.00, fee=0.0006):
        self.cash = initial_capital
        self.fee = fee # defaulted to 0.06%
        self.shares = 0.0
        self.portfolio_value = initial_capital
        self.equity_curve = []

    def run(self, data, strategy):
        
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['close']

            target_pct = strategy.on_data(row)

            self.rebalance(target_pct, price, row['time'])

            total_value = self.cash + (self.shares*price)
            self.equity_curve.append({'time': row['time'], 'value': total_value})
    
    def rebalance(self, target_pct, price, time):
        cur_total_val = self.cash + (self.shares * price)
        target_holding_val = cur_total_val * target_pct

        cur_holding_val = self.shares * price
        diff_value = target_holding_val - cur_holding_val

        if abs(diff_value) < 5.0:
            return
        
        # buy more
        if diff_value > 0:
            qty_to_buy = diff_value/price
            if self.cash >= diff_value:
                self.shares += qty_to_buy
                self.cash -= diff_value * (1 + self.fee)
                print(f"[{time}] BUY: {qty_to_buy:.4f} shares @ ${price:.2f}")
            elif self.cash > 0:
                # Calculate max buyable amount: Cash = Amount * (1 + Fee)
                # Amount = Cash / (1 + Fee)
                max_buy_val = self.cash / (1 + self.fee)
                qty_to_buy = max_buy_val / price
                
                self.shares += qty_to_buy
                self.cash = 0 # Spent it all
                print(f"[{time}] BUY (Max): {qty_to_buy:.4f} shares @ ${price:.2f}")
            else:
                print("Not enough cash")
        else: # sell off
            qty_to_sell = abs(diff_value)/price
            self.shares -= qty_to_sell
            self.cash += abs(diff_value) * (1-self.fee)
            print(f"[{time}] SELL: {qty_to_sell:.4f} shares @ ${price:.2f}")
        
