import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
# 1. DYNAMIC TRIPLE BARRIER CONFIG (1:2 Risk/Reward)
VERTICAL_BARRIER = 12       # Maximum time to hold the trade (candles)
ATR_WINDOW = 14             # Standard period for ATR calculation
ATR_MULTIPLIER_TP = 2.0     # Take Profit = 2x the ATR percentage
ATR_MULTIPLIER_SL = 1.0     # Stop Loss = 1x the ATR percentage

# 2. VAE CONFIG (The Safety Filter)
VAE_BETA_STD = 3.0         
VAE_WINDOW = 30            

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------
# 1. MODEL DEFINITION
# ---------------------------------------------------------
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

def load_vae_model(input_dim):
    config = joblib.load("MLP_VAE_Strategy/vae_config.pkl")
    model = MarketVAE(
        input_dim=input_dim,
        hidden_1=config['hidden_1'],
        hidden_2=config['hidden_2'],
        latent_dim=config['latent_dim']
    ).to(DEVICE)
    model.load_state_dict(torch.load("MLP_VAE_Strategy/model&scalers&configs/vae_model_mlp.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    return model

# ---------------------------------------------------------
# 2. HYBRID LABELING LOGIC (DYNAMIC ATR BARRIERS)
# ---------------------------------------------------------
def generate_hybrid_labels(df, model, feature_cols):
    print("  -> Sorting and grouping data...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if 'symbol' in df.columns:
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        group_col = 'symbol'
    else:
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['dummy_symbol'] = 'ASSET'
        group_col = 'dummy_symbol'

    # ---------------------------------------------------------
    # 3. CALCULATE ATR AS A PERCENTAGE
    # ---------------------------------------------------------
    print("  -> Calculating Dynamic Volatility (ATR%)...")
    df['prev_close'] = df.groupby(group_col)['close'].shift(1)
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df.groupby(group_col)['tr'].transform(lambda x: x.rolling(window=ATR_WINDOW).mean())
    df['atr_pct'] = df['atr'] / df['close']

    # ---------------------------------------------------------
    # 4. VAE ERROR CALCULATION
    # ---------------------------------------------------------
    print("  -> Calculating VAE Safety Filter...")
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(features)
        loss_per_row = F.mse_loss(recon, features, reduction='none').mean(dim=1)
    
    df['vae_error'] = loss_per_row.cpu().numpy()
    grouper = df.groupby(group_col)['vae_error']
    rolling_mean = grouper.transform(lambda x: x.rolling(window=VAE_WINDOW).mean())
    rolling_std = grouper.transform(lambda x: x.rolling(window=VAE_WINDOW).std())
    
    df['vae_threshold'] = rolling_mean + (VAE_BETA_STD * rolling_std)
    df['vae_threshold'] = df['vae_threshold'].bfill() 

    # ---------------------------------------------------------
    # 5. VECTORIZED TRIPLE BARRIER METHOD (DYNAMIC + WICKS)
    # ---------------------------------------------------------
    print("  -> Computing Dynamic Triple Barrier Paths (High/Low Wicks)...")
    
    fut_high_rets = []
    fut_low_rets = []
    for i in range(1, VERTICAL_BARRIER + 1):
        # Highs for checking Take Profit (Longs) / Stop Loss (Shorts)
        high_ret = (df.groupby(group_col)['high'].shift(-i) / df['close']) - 1.0
        # Lows for checking Stop Loss (Longs) / Take Profit (Shorts)
        low_ret = (df.groupby(group_col)['low'].shift(-i) / df['close']) - 1.0
        
        fut_high_rets.append(high_ret.values)
        fut_low_rets.append(low_ret.values)
        
    high_rets_matrix = np.column_stack(fut_high_rets)
    low_rets_matrix = np.column_stack(fut_low_rets)
    
    tp_thresholds = (df['atr_pct'] * ATR_MULTIPLIER_TP).values[:, None]
    sl_thresholds = (df['atr_pct'] * ATR_MULTIPLIER_SL).values[:, None]
    
    # --- BUY LOGIC (LONG) ---
    tp_mask_buy = high_rets_matrix >= tp_thresholds
    sl_mask_buy = low_rets_matrix <= -sl_thresholds  # Negative sign for drop
    
    tp_idx_buy = np.argmax(tp_mask_buy, axis=1)
    sl_idx_buy = np.argmax(sl_mask_buy, axis=1)
    
    tp_idx_buy = np.where(np.any(tp_mask_buy, axis=1), tp_idx_buy, 999)
    sl_idx_buy = np.where(np.any(sl_mask_buy, axis=1), sl_idx_buy, 999)
    
    # Strict inequality (<): If TP and SL are hit in the exact same candle, 
    # we assume the Stop Loss hit first to be defensively pessimistic.
    is_pump = tp_idx_buy < sl_idx_buy
    
    # --- SELL LOGIC (SHORT) ---
    tp_mask_sell = low_rets_matrix <= -tp_thresholds
    sl_mask_sell = high_rets_matrix >= sl_thresholds 
    
    tp_idx_sell = np.argmax(tp_mask_sell, axis=1)
    sl_idx_sell = np.argmax(sl_mask_sell, axis=1)
    
    tp_idx_sell = np.where(np.any(tp_mask_sell, axis=1), tp_idx_sell, 999)
    sl_idx_sell = np.where(np.any(sl_mask_sell, axis=1), sl_idx_sell, 999)
    
    is_dump = tp_idx_sell < sl_idx_sell

    # ---------------------------------------------------------
    # 6. APPLY LABELS & CLEANUP
    # ---------------------------------------------------------
    df['mlp_label'] = 0 
    is_safe = df['vae_error'] < df['vae_threshold']
    
    df.loc[is_safe & is_pump, 'mlp_label'] = 1
    df.loc[is_safe & is_dump, 'mlp_label'] = 2
    
    valid_mask = ~np.isnan(high_rets_matrix[:, -1]) & ~df['atr_pct'].isna()
    df = df[valid_mask].copy()
    
    base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    cols_to_keep = [c for c in base_cols if c in df.columns] + feature_cols + ['mlp_label']
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    
    df = df[cols_to_keep].copy()
        
    print(f"  -> Final Label Dist: {df['mlp_label'].value_counts().to_dict()}")
    return df

# ---------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("Loading data...")
    train_df = pd.read_csv("MLP_VAE_Strategy/train_val_test_datasets/train_data.csv")
    val_df   = pd.read_csv("MLP_VAE_Strategy/train_val_test_datasets/val_data.csv")
    test_df  = pd.read_csv("MLP_VAE_Strategy/train_val_test_datasets/test_data.csv")
    
    feature_cols = joblib.load("MLP_VAE_Strategy/feature_columns.pkl")
    input_dim = len(feature_cols)
    print(f"Loaded Feature Map: {input_dim} columns expected.")
    
    print("Loading VAE Model...")
    model = load_vae_model(input_dim)
    
    print("\n--- Processing TRAIN ---")
    train_labeled = generate_hybrid_labels(train_df, model, feature_cols)
    
    print("\n--- Processing VAL ---")
    val_labeled = generate_hybrid_labels(val_df, model, feature_cols)
    
    print("\n--- Processing TEST ---")
    test_labeled = generate_hybrid_labels(test_df, model, feature_cols)
    
    train_labeled.to_csv("MLP_VAE_Strategy/train_val_test_datasets/train_labeled.csv", index=False)
    val_labeled.to_csv("MLP_VAE_Strategy/train_val_test_datasets/val_labeled.csv", index=False)
    test_labeled.to_csv("MLP_VAE_Strategy/train_val_test_datasets/test_labeled.csv", index=False)
    
    print("\nProcessing Complete. Dynamic ATR TBM Labels Generated!")

if __name__ == "__main__":
    main()