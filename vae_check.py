import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from UL_RL_strategy import *
import joblib

# Assume RollingWindowDataset and LSTM_VAE are already defined/imported

model_path = 'vae_model_epoch7.pth'

def verify_reconstruction(dataset, scaler):
    # 1. Load Model
    # Note: Ensure LSTM_VAE class is defined before running this!
    model = torch.load(model_path)
    model.eval()
    device = next(model.parameters()).device
    
    # 2. Get a Sample
    # We take a specific index (100) or random one
    idx = 100 
    sample_scaled_tensor = dataset[idx].unsqueeze(0).to(device) # Shape (1, 64, 4)
    
    # 3. Run Inference
    with torch.no_grad():
        recon_scaled_tensor, _, _, _ = model(sample_scaled_tensor)
    
    # 4. INVERSE TRANSFORM (The Critical Fix for Visualization)
    # Convert tensors to numpy arrays
    sample_scaled_np = np.array(sample_scaled_tensor.cpu().tolist())[0]
    recon_scaled_np = np.array(recon_scaled_tensor.cpu().tolist())[0]
    
    # Use the scaler to turn Z-scores back into Real Values
    sample_real = scaler.inverse_transform(sample_scaled_np)
    recon_real = scaler.inverse_transform(recon_scaled_np)
    
    # 5. Plot Feature 0 (Log Returns)
    # Slicing [:, 0] gets all 64 time steps for the 0th feature
    orig_series = sample_real[:, 0]
    recon_series = recon_real[:, 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(orig_series, label="Original (Real Units)", color='tab:blue')
    plt.plot(recon_series, label="Reconstruction (Real Units)", color='tab:orange', linestyle='--')
    plt.title(f"VAE Reconstruction Check - Index {idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- DATA PREPARATION ---
data = df[['log_return', 'relative_volume', 'volatility', 'rsi']]
data = data.iloc[64:]

# 1. Handle Infinite values BEFORE scaling (Safety Check)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

# 2. Scale the data
scaler = joblib.load('vae_scaler.pkl')
lower = data.quantile(0.01)
upper = data.quantile(0.99)
data = data.clip(lower=lower, upper=upper, axis=1)

# Transform using the LOADED scaler
scaled_data = scaler.transform(data.values)

# 3. FIX: Pass 'scaled_data', not 'data'
dataset = RollingWindowDataset(scaled_data, window_size=64)

# 4. Run verification passing the SCALER too

verify_reconstruction(dataset, scaler)

