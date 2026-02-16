import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import itertools
import copy
import joblib

# --- CONFIGURATION ---
BATCH_SIZE = 1024
EPOCHS_PER_TRIAL = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BETA = 0.5

# --- GRID ---
param_grid = {
    'hidden_1': [48, 64],
    'hidden_2': [32],
    'latent_dim': [6],
    'lr': [1e-3, 5e-4]
}

# ---------------------------------------------------------
# 1. MODEL
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
        self.dec_output = nn.Linear(hidden_1, input_dim) # Linear output is correct for Scaled Data

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

# ---------------------------------------------------------
# 2. LOSS
# ---------------------------------------------------------
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + (beta * KLD)

# ---------------------------------------------------------
# 3. UPDATED DATA LOADER
# ---------------------------------------------------------
def load_data(filepath, feature_cols):
    """
    Loads data and filters ONLY the features used in the Split step.
    This ensures VAE and MLP see the exact same world.
    """
    df = pd.read_csv(filepath)
    
    # 1. Verify all columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")
    
    # 2. Select ONLY the feature columns (Automatically drops label, symbol, etc.)
    features = df[feature_cols].values
    
    return torch.tensor(features, dtype=torch.float32)

# ---------------------------------------------------------
# 4. TRAIN CANDIDATE
# ---------------------------------------------------------
def train_candidate(config, train_loader, val_loader, input_dim):
    model = MarketVAE(input_dim, config['hidden_1'], config['hidden_2'], config['latent_dim']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model.train()
    
    for epoch in range(EPOCHS_PER_TRIAL):
        for data in train_loader:
            x = data[0].to(DEVICE)
            recon, mu, log_var = model(x)
            loss = vae_loss_function(recon, x, mu, log_var, beta=BETA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Validation
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for data in val_loader:
            x = data[0].to(DEVICE)
            recon, mu, log_var = model(x)
            val_loss_sum += vae_loss_function(recon, x, mu, log_var, beta=BETA).item()
            
    return val_loss_sum / len(val_loader.dataset), model.state_dict()

# ---------------------------------------------------------
# 5. MAIN (Updated)
# ---------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. LOAD THE MAP (The Source of Truth)
    # This file was created by your Split script. It contains the list of 36ish features.
    feature_cols = joblib.load("MLP_VAE_Strategy/feature_columns.pkl")
    print(f"Loaded Feature Map: {len(feature_cols)} columns")

    # 2. LOAD DATA
    print("Loading data...")
    train_data = load_data("MLP_VAE_Strategy/train_data.csv", feature_cols)
    val_data = load_data("MLP_VAE_Strategy/val_data.csv", feature_cols)
    
    input_dim = train_data.shape[1]
    print(f"Data Loaded. Input Features: {input_dim}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. RUN TOURNAMENT
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_val_loss = float('inf')
    best_config = None
    best_model_state = None
    
    print(f"\nStarting Tournament with {len(combinations)} models...")
    print(f"{'ID':<4} | {'H1':<4} | {'H2':<4} | {'Latent':<6} | {'LR':<8} | {'Val Loss'}")
    print("-" * 55)
    
    for i, config in enumerate(combinations):
        val_loss, model_state = train_candidate(config, train_loader, val_loader, input_dim)
        print(f"{i+1:<4} | {config['hidden_1']:<4} | {config['hidden_2']:<4} | {config['latent_dim']:<6} | {config['lr']:<8} | {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            best_model_state = copy.deepcopy(model_state)
            
    # 4. SAVE CHAMPION
    print("-" * 55)
    print(f"WINNER: {best_config}")
    print(f"Lowest Val Loss: {best_val_loss:.4f}")
    
    # Save Model Weights
    final_model = MarketVAE(input_dim, best_config['hidden_1'], best_config['hidden_2'], best_config['latent_dim'])
    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), "MLP_VAE_Strategy/vae_model_mlp.pth")
    
    # Save Config Dictionary (Need this for Backtesting!)
    joblib.dump(best_config, "MLP_VAE_Strategy/vae_config.pkl") 
    print("\nSaved: vae_model_mlp.pth & vae_config.pkl")

if __name__ == "__main__":
    main()