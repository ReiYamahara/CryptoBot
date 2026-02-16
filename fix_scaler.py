import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIG ---
RAW_DATA_PATH = "MLP_VAE_Strategy/features_dataset.csv" 
LABELED_DATA_PATH = "MLP_VAE_Strategy/train_labeled.csv"

# Output Paths
TECH_SCALER_PATH = "MLP_VAE_Strategy/tech_scaler.pkl"
VAE_SCALER_PATH = "MLP_VAE_Strategy/vae_scaler.pkl"
COLUMNS_PATH = "MLP_VAE_Strategy/feature_columns.pkl"

# Lockout settings (Must match training)
LOCKOUT_SYMBOLS = ["AAVEUSD", "LSK", "REPUSD"] 
VAL_START_DATE = pd.Timestamp("2025-01-01").tz_localize('UTC')

def main():
    print("--- GENERATING SPLIT SCALERS ---")
    
    # 1. Create Technical Scaler (from Raw Data)
    print(f"1. Processing Technicals from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True)
    df_raw.sort_values('timestamp', inplace=True)
    
    # Filter for Training Set
    df_main = df_raw[~df_raw['symbol'].isin(LOCKOUT_SYMBOLS)]
    train_set = df_main[df_main['timestamp'] < VAL_START_DATE]
    
    # Identify Columns
    cols_to_exclude = ['timestamp', 'symbol', 'label', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vae_error', 'mlp_label', 'future_return', 'Unnamed: 0']
    tech_cols = [c for c in train_set.columns if c not in cols_to_exclude]
    
    print(f"   Identified {len(tech_cols)} technical features.")
    
    # Fit & Save Tech Scaler
    tech_scaler = StandardScaler()
    tech_scaler.fit(train_set[tech_cols])
    joblib.dump(tech_scaler, TECH_SCALER_PATH)
    
    # Save Column Order (Crucial for Strategy)
    joblib.dump(tech_cols, COLUMNS_PATH)
    print(f"   Saved {TECH_SCALER_PATH}")
    print(f"   Saved {COLUMNS_PATH} (Ensures column order matches)")

    # 2. Create VAE Scaler (from Labeled Data)
    print(f"\n2. Processing VAE Error from {LABELED_DATA_PATH}...")
    df_labeled = pd.read_csv(LABELED_DATA_PATH)
    
    if 'vae_error' not in df_labeled.columns:
        raise ValueError("vae_error column missing!")
        
    # Fit & Save VAE Scaler
    vae_scaler = StandardScaler()
    # Reshape because fit expects 2D array
    vae_scaler.fit(df_labeled[['vae_error']])
    
    joblib.dump(vae_scaler, VAE_SCALER_PATH)
    print(f"   Saved {VAE_SCALER_PATH}")
    print(f"   VAE Mean: {vae_scaler.mean_[0]:.6f}, Scale: {vae_scaler.scale_[0]:.6f}")
    
    print("\nDONE. Update your Strategy to use these two scalers.")

if __name__ == "__main__":
    main()