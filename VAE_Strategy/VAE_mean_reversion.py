from VAE_Autoencoder import *
import joblib
import pandas_ta as ta

def preprocess_with_scaling(df):
    """
    Args:
        df: The DataFrame returned by fetch_live_data()
    """
    df = df.copy()

    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Feature Engineering ---

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['sma_volume'] = df['volume'].rolling(window=64).mean()
    df['relative_volume'] = df['volume'] / df['sma_volume']
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['rsi'] = calculate_rsi(df)

    # --- Filter & Clean ---
    data = df[['log_return', 'relative_volume', 'volatility', 'rsi']]
    
    # Remove the first 64 rows (which are NaN due to the rolling window)
    data = data.iloc[64:]

    # --- Scale ---
    # Ensure 'vae_scaler.pkl' is in your project folder
    scaler = joblib.load('vae_scaler.pkl')
    scaled_data = scaler.transform(data.values)
    
    return scaled_data

class SmartMeanReversion:
    # Need 64 (drop) + 64 (VAE window) = 128 bars before we can run
    MIN_BARS = 200

    def __init__(self, vae_model, scaler, device, risk_threshold=4.0):
        self.vae = vae_model
        self.scaler = scaler
        self.device = device
        self.threshold = risk_threshold
        self._buffer = pd.DataFrame()

        self.position_target = 0.0 # 0.0 = Cash, 1.0 = Invested
        self.entry_price = 0.0
        
        # --- STRATEGY SETTINGS ---
        self.stop_loss_pct = 0.03    # 3% Max Loss
        self.take_profit_pct = 0.05  # 5% Target Profit
        self.ema_period = 200        # Trend Filter
        self.bb_length = 20          # Bollinger Band Length
        self.bb_std = 2.0            # Bollinger Band Deviation

    def reset(self):
        """Clear history (e.g. before a new backtest run)."""
        self._buffer = pd.DataFrame()

    def get_anomaly_score(self, df):
        scaled_data = preprocess_with_scaling(df)
        input_tensor = torch.tensor(scaled_data[-64:], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, _, _, _ = self.vae(input_tensor)
        loss = torch.nn.functional.mse_loss(recon, input_tensor)
        return loss.item()

    def on_data(self, row):
        # Accept one row at a time (Series or 1-row DataFrame), like live
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        self._buffer = pd.concat([self._buffer, row], ignore_index=True)

        # Optional: cap buffer size for long-running live
        if len(self._buffer) > 200:
            self._buffer = self._buffer.iloc[-200:]

        if len(self._buffer) < self.MIN_BARS:
            return 0.0

        # Run on full history buffer
        score = self.get_anomaly_score(self._buffer)

        if score > self.threshold:
            print(f"[RISK ALERT] Anomaly Score {score:.4f} is too high! Market is chaotic. Staying Cash.")
            return 0.0

        print(f"[SAFE] Anomaly Score {score:.4f} is normal. Running logic...")

        df = self._buffer.copy()
        close_prices = df['close'].astype(float)
        
        # 1. EMA 200 (Trend Filter)
        df['ema_200'] = ta.ema(close_prices, length=self.ema_period)
        
        # 2. Bollinger Bands
        bb = ta.bbands(close_prices, length=self.bb_length, std=self.bb_std)
        bb.columns = ['LOWER', 'MID', 'UPPER', 'BANDWIDTH', 'PERCENT']
        df['bb_lower'] = bb['LOWER']
        df['bb_upper'] = bb['UPPER']
        df['bb_mid']   = bb['MID']
        
        # 3. RSI
        df['rsi'] = ta.rsi(close_prices, length=14)

        # Get the latest candle (current moment)
        current = df.iloc[-1]
        current_price = current['close'].astype(float)

        # --- D. Portfolio Management (Stop Loss / Take Profit) ---
        if self.position_target > 0:
            pct_change = (current_price - self.entry_price) / self.entry_price
            
            # Stop Loss
            if pct_change < -self.stop_loss_pct:
                print(f"[EXIT] Stop Loss Hit! {pct_change:.2%}")
                self.position_target = 0.0
                return 0.0
            
            # Take Profit
            if pct_change > self.take_profit_pct:
                print(f"[EXIT] Take Profit Hit! {pct_change:.2%}")
                self.position_target = 0.0
                return 0.0

        # --- E. Entry Logic (Buy) ---
        # CONDITION 1: We are currently in Cash
        if self.position_target == 0.0:
            # CONDITION 2: Market is in an UPTREND (Price > EMA 200)
            # (Prevent catching falling knives in a bear market)
            if current_price > current['ema_200']:
                
                # CONDITION 3: Price crashed below Lower Bollinger Band
                # CONDITION 4: RSI is Oversold (< 35)
                if current_price < current['bb_lower'] and current['rsi'] < 35:
                    print(f"[ENTRY] Dip Detected in Uptrend. Price: {current_price:.2f}, RSI: {current['rsi']:.1f}")
                    self.position_target = 1.0
                    self.entry_price = current_price
                    return 1.0

        # --- F. Exit Logic (Sell) ---
        # CONDITION 1: We are currently Invested
        elif self.position_target > 0.0:
            # CONDITION 2: Price Reverted to the Mean (Middle Band)
            # We don't wait for RSI 70. We take profit at the average.
            if current_price > current['bb_mid']:
                print(f"[EXIT] Mean Reversion Complete. Price hit Middle Band.")
                self.position_target = 0.0
                return 0.0
                
        # Return whatever our current target is (Hold 0.0 or Hold 1.0)
        return self.position_target