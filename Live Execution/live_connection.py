import ccxt
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime
import os
from dotenv import load_dotenv
import math

load_dotenv()

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_PRIVATE_KEY = os.getenv("KRAKEN_PRIVATE_KEY")

# 1. SETUP KRAKEN CONNECTION
exchange = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_PRIVATE_KEY,
    'enableRateLimit': True,
})

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=8, seq_len=64):
        super(LSTM_VAE, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared encoder backbone
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True,
            bidirectional=True
        )

        # Heads for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)

        # Decoder
        self.decoder_project = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

        self.final_layer = nn.Linear(hidden_dim, input_dim)


    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n_concatenated = torch.cat((h_n[0], h_n[1]), dim=1)
        
        # 2. Get distribution parameters
        mu = self.fc_mu(h_n_concatenated)
        logvar = self.fc_logvar(h_n_concatenated)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_projected = self.decoder_project(z)
        
        # 2. "Repeat" z to create a sequence for the LSTM
        # We want input shape: (Batch, seq_len, hidden_dim)
        # So we repeat the vector 64 times
        decoder_input = z_projected.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 3. Pass through Decoder LSTM
        # out shape: (Batch, seq_len, hidden_dim)
        out, _ = self.decoder_lstm(decoder_input)
        
        # 4. Map back to features (Batch, seq_len, 4)
        x_recon = self.final_layer(out)
        
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

# 2. LOAD RESOURCES
# Ensure your LSTM_VAE class is defined in this file or imported!
device = torch.device("cpu") # CPU is fine for inference
model = torch.load('vae_model.pth', map_location=device)
model.eval()
scaler = joblib.load('vae_scaler.pkl')

# Configuration
SYMBOL = 'BTC/USDT'  # Kraken symbol format
TIMEFRAME = '1m'     # Must match your training data
WINDOW_SIZE = 64     # The sequence length your model expects

def calculate_rsi(df, period=14):
    # Calculate the price change
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate Wilder's Smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_live_data(symbol, timeframe, limit=100):
    """
    Fetches OHLCV data from Kraken
    """
    # Fetch slightly more than needed to calculate indicators (RSI needs warm-up)
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def preprocess_live_data(df):
    """
    Applies EXACTLY the same steps as your training script
    """
    # 1. Feature Engineering
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['sma_volume'] = df['volume'].rolling(window=64).mean()
    df['relative_volume'] = df['volume'] / df['sma_volume']
    df['volatility'] = (df['high'] - df['low']) / df['close']
    
    # Calculate RSI (Paste your RSI function here)
    df['rsi'] = calculate_rsi(df) # Ensure this function exists
    
    # 2. Select Features
    features = df[['log_return', 'relative_volume', 'volatility', 'rsi']].copy()
    
    # 3. Clean (Handle NaNs created by rolling windows)
    features.fillna(0, inplace=True)
    
    # 4. CRITICAL: CLIP OUTLIERS
    # Ideally, hardcode the values you found during training to be safe
    # e.g., lower = -0.05, upper = 0.05
    # If using dynamic quantile on small live window, it might be unstable.
    # BEST PRACTICE: Save your training lower/upper limits to a config and load them here.
    lower = features.quantile(0.01) 
    upper = features.quantile(0.99)
    features = features.clip(lower=lower, upper=upper, axis=1)
    
    # 5. Scale using the LOADED scaler
    scaled_data = scaler.transform(features)
    
    return scaled_data

def wait_for_next_minute():
    """
    Sleeps exactly until the start of the next minute + 1 second buffer.
    """
    now = time.time()
    # Calculate time to next full minute
    next_minute = math.ceil(now / 60) * 60
    
    # Add 1-2 seconds buffer to ensure Kraken has finished writing the candle
    sleep_seconds = next_minute - now + 2 
    
    if sleep_seconds < 0:
        sleep_seconds += 60
        
    print(f"Waiting {sleep_seconds:.2f}s for next candle...")
    time.sleep(sleep_seconds)

# --- MAIN LOOP ---
print("Bot started...")
while True:
    try:
        # A. Get Data
        df = fetch_live_data(SYMBOL, TIMEFRAME, limit=200)
        
        # B. Prepare the last window
        scaled_data = preprocess_live_data(df)
        
        # Grab the last 64 rows for the model
        current_window = scaled_data[-WINDOW_SIZE:] 
        
        # Check if we have enough data
        if len(current_window) < WINDOW_SIZE:
            print("Not enough data yet...")
            time.sleep(60)
            continue
        
        current_window = current_window.astype(np.float32)

        # C. Inference
        tensor_window = torch.from_numpy(current_window).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, mu, logvar, z = model(tensor_window)
            
        # D. Trading Logic (Example)
        # Calculate reconstruction error (Anomaly Score)
        recon_np = recon.detach().cpu().numpy()[0]
        mse = np.mean((current_window - recon_np)**2)
        
        print(f"Time: {datetime.now()} | Anomaly Score: {mse:.6f}")
        
        # E. Execution Logic (Pseudo-code)
        # if mse > threshold:
        #     order = exchange.create_market_buy_order(SYMBOL, amount)
        #     print("BUY EXECUTED")
        
        # F. Sleep
        # Sleep for 1 hour (or whatever your timeframe is)
        # Better: Sleep until the next candle close
        time.sleep(10) 

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(60) # Wait a minute before retrying