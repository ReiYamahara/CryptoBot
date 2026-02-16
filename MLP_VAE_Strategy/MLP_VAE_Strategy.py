import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
import torch.nn.functional as F  # Added for softmax/mse_loss

# --- KEEP YOUR MODEL CLASSES AS IS ---
class MarketVAE(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, latent_dim):
        super(MarketVAE, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_1)
        self.enc2 = nn.Linear(hidden_1, hidden_2)
        self.z_mean = nn.Linear(hidden_2, latent_dim)
        self.z_log_var = nn.Linear(hidden_2, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_2)
        self.dec2 = nn.Linear(hidden_2, hidden_1)
        self.dec_output = nn.Linear(hidden_1, input_dim)
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = torch.tanh(self.enc1(x))
        h = torch.tanh(self.enc2(h))
        mu = self.z_mean(h)
        log_var = self.z_log_var(h)
        z = self.reparameterize(mu, log_var)
        h_dec = torch.tanh(self.dec1(z))
        h_dec = torch.tanh(self.dec2(h_dec))
        reconstruction = self.dec_output(h_dec)
        return reconstruction, mu, log_var

class TradingMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TradingMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # <--- Missing layer 1
        
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)   # <--- Missing layer 2
        
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)   # <--- Missing layer 3
        
        self.output = nn.Linear(32, num_classes)
        self.leaky_relu = nn.LeakyReLU() 
        
    def forward(self, x):
        # Apply Linear -> BatchNorm -> Activation
        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.leaky_relu(self.bn3(self.layer3(x)))
        return self.output(x)

class MLPStrategy:
    def __init__(self, mlp_path, vae_path, vae_config, threshold=0.74, device="cpu"):
        self.device = device
        self.threshold = threshold 
        
        # DYNAMIC ATR MULTIPLIERS (1:2 R:R)
        self.atr_mult_tp = 2.0 
        self.atr_mult_sl = 1.0
        
        print(f"--- INITIALIZING STRATEGY (SINGLE SCALER MODE) ---")
        
        # 1. LOAD SINGLE SCALER & COLUMN MAP
        self.scaler = joblib.load("MLP_VAE_Strategy/std_scaler.pkl")
        self.tech_cols = joblib.load("MLP_VAE_Strategy/feature_columns.pkl")
        
        print(f"1. Standard Scaler Loaded.")
        print(f"2. Strategy expects {len(self.tech_cols)} Technical Features.")

        # 2. Load Models
        self.vae = MarketVAE(
            input_dim=len(self.tech_cols), 
            hidden_1=vae_config['hidden_1'], 
            hidden_2=vae_config['hidden_2'], 
            latent_dim=vae_config['latent_dim']
        ).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device, weights_only=True))
        self.vae.eval()
        
        # IMPORTANT: Input dim is exactly the number of tech cols
        self.mlp = TradingMLP(input_dim=len(self.tech_cols)).to(self.device)
        self.mlp.load_state_dict(torch.load(mlp_path, map_location=self.device, weights_only=True))
        self.mlp.eval()
        
        self.buffer = pd.DataFrame()
        self.min_bars_needed = 200 
        
        # Position State Tracking
        self.current_position = 0.0
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None

    def derive_features(self, df):
        df = df.copy()
        
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(float)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, unit='s')

        df.sort_index(inplace=True)

        df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(4))
        df['log_ret_12h'] = np.log(df['close'] / df['close'].shift(12))
        df['log_ret_24h'] = np.log(df['close'] / df['close'].shift(24))

        df['rsi_14'] = ta.rsi(df['close'], length=14) / 100.0
        df['rsi_7'] = ta.rsi(df['close'], length=7) / 100.0
        
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['stoch_k'] = stoch.iloc[:, 0] / 100.0 
            df['stoch_d'] = stoch.iloc[:, 1] / 100.0
        else:
            df['stoch_k'] = 0; df['stoch_d'] = 0

        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20) / 1000.0
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14) / -100.0

        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bb_pct'] = bbands.iloc[:, -1] 
            df['bb_width'] = bbands.iloc[:, 3]
        
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
        df['std_20'] = df['close'].rolling(20).std() / df['close']
        df['donchian_width'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
        df['return_vol_ratio'] = df['log_ret_1h'].rolling(24).mean() / (df['log_ret_1h'].rolling(24).std() + 1e-9)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None: df['adx'] = adx_df.iloc[:, 0] / 100.0
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd_line'] = macd.iloc[:, 0] / df['close']
            df['macd_hist'] = macd.iloc[:, 1] / df['close']
        
        aroon = ta.aroon(df['high'], df['low'], length=25)
        if aroon is not None:
            df['aroon_up'] = aroon.iloc[:, 0] / 100.0
            df['aroon_down'] = aroon.iloc[:, 1] / 100.0
            df['aroon_osc'] = aroon.iloc[:, 2] / 100.0

        for length in [7, 25, 99, 200]:
            sma = ta.sma(df['close'], length=length)
            ema = ta.ema(df['close'], length=length)
            if sma is not None: df[f'dist_sma_{length}'] = (df['close'] / sma) - 1
            if ema is not None: df[f'dist_ema_{length}'] = (df['close'] / ema) - 1

        df['vol_change_1'] = np.log(df['volume'] / df['volume'].shift(1).replace(0, 1))
        df['rel_vol_24'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-9)
        
        obv = ta.obv(df['close'], df['volume'])
        if obv is not None: df['obv_norm'] = obv / (df['volume'].rolling(200).sum() + 1e-9)
        
        df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14) / 100.0
        
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap is not None: df['dist_vwap'] = (df['close'] / vwap) - 1
        else: df['dist_vwap'] = 0

        eom = ta.eom(df['high'], df['low'], df['close'], df['volume'], length=14)
        if eom is not None: df['eom_squashed'] = np.tanh(eom / 1e6)
        
        df.dropna(inplace=True)
        return df

    def on_data(self, row):
        # 1. Ingestion
        if isinstance(row, pd.Series): row = row.to_frame().T
        
        current_close = float(row['close'].iloc[0])
        current_high  = float(row['high'].iloc[0])
        current_low   = float(row['low'].iloc[0])
        current_time  = pd.to_datetime(row['timestamp'].iloc[0], unit='s', utc=True)
        
        self.buffer = pd.concat([self.buffer, row], ignore_index=True)
        # Give ourselves enough buffer room for 200 SMA + some buffer
        if len(self.buffer) > 1000: self.buffer = self.buffer.iloc[-1000:]

        # -----------------------------------------------------------------
        # 2. STRICT BARRIER EXIT LOGIC (LONG ONLY)
        # -----------------------------------------------------------------
        if self.current_position > 0.0:
            if current_low <= self.sl_price or current_high >= self.tp_price:
                self.current_position = 0.0
                self.entry_price = self.tp_price = self.sl_price = None
                return 0.0
            
            return self.current_position

        if len(self.buffer) < self.min_bars_needed: return 0.0 

        # -----------------------------------------------------------------
        # 3. ENTRY LOGIC
        # -----------------------------------------------------------------
        if self.current_position == 0.0:
            
            full_df = self.derive_features(self.buffer)
            if len(full_df) == 0: return 0.0
            
            # A. Extract Raw Technical Features
            try:
                raw_tech = full_df[self.tech_cols].iloc[-1].values.reshape(1, -1)
            except KeyError as e:
                print(f"🚨 FATAL ERROR: Missing Feature Column: {e}")
                return 0.0
            
            # B. Scale using the single StandardScaler
            scaled_tech = self.scaler.transform(raw_tech)
            
            # (Optional) You can still run the VAE here if you want to log it,
            # but it's no longer forced into the MLP since the MLP was trained
            # strictly on the technical features.
            
            # C. Run MLP
            tensor_mlp_in = torch.tensor(scaled_tech, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = self.mlp(tensor_mlp_in)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                
            prob_hold, prob_buy, prob_sell = probs[0], probs[1], probs[2]


            # Grab current ATR Percentage from our feature dataframe
            current_atr_pct = float(full_df['atr_14'].iloc[-1])
            
            # -----------------------------------------------------------------
            # 4. SIGNAL GENERATION (LONG ONLY)
            # -----------------------------------------------------------------
            # Must beat our optimal threshold AND beat the Sell probability
            if prob_buy > self.threshold and prob_buy >= prob_sell:
                self.current_position = 1.0
                self.entry_price = current_close
                
                # Lock in the physical price barriers based on volatility at entry
                self.tp_price = current_close * (1 + (current_atr_pct * self.atr_mult_tp))
                self.sl_price = current_close * (1 - (current_atr_pct * self.atr_mult_sl))
                
                print(f"📈 BUY SIGNAL: {current_time} | Prob: {prob_buy:.2f} | Entry: {current_close:.2f} | TP: {self.tp_price:.2f} | SL: {self.sl_price:.2f}")
                return 1.0

        return self.current_position