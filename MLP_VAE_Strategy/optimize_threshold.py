import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
VAL_FILE = "MLP_VAE_Strategy/val_labeled.csv"
MODEL_FILE = "MLP_VAE_Strategy/mlp_model_best.pth"
FEATURE_MAP = "MLP_VAE_Strategy/feature_columns.pkl"

# --- NEW FILTERS ---
TARGET_SYMBOL = "BTCUSDT"  # Change this to match your CSV (e.g., 'XBTUSDT' or 'BTC')
THRESHOLDS = np.arange(0.36, 0.78, 0.02)

# ---------------------------------------------------------
# 1. MODEL DEFINITION (Must match training exact architecture)
# ---------------------------------------------------------
class TradingMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TradingMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, num_classes)
        self.leaky_relu = nn.LeakyReLU() 

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.leaky_relu(self.bn3(self.layer3(x)))
        return self.output(x)

# ---------------------------------------------------------
# 2. EVALUATION LOGIC
# ---------------------------------------------------------
def main():
    print("Loading feature map...")
    feature_cols = joblib.load(FEATURE_MAP)
    input_dim = len(feature_cols)
    
    print("Loading validation data...")
    val_df = pd.read_csv(VAL_FILE)
    
    # --- FILTER FOR BITCOIN ONLY ---
    if 'symbol' in val_df.columns:
        print(f"Filtering dataset for {TARGET_SYMBOL} only...")
        val_df = val_df[val_df['symbol'] == TARGET_SYMBOL].reset_index(drop=True)
        if len(val_df) == 0:
            print(f"WARNING: No rows found for symbol {TARGET_SYMBOL}! Check your exact ticker name.")
            return
    else:
        print("No 'symbol' column found. Assuming dataset is already single-asset.")

    print(f"Evaluating on {len(val_df)} rows...")
    
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    y_val = val_df['mlp_label'].values  # 0=Hold, 1=Buy, 2=Sell
    
    print("Loading trained model...")
    model = TradingMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 1. Generate Probabilities
    print("Extracting Softmax probabilities...")
    with torch.no_grad():
        outputs = model(X_val)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
    results = []
    
    # 2. Iterate through grid
    print(f"\nRunning Buy-Only Grid Search across {len(THRESHOLDS)} thresholds...\n")
    
    holding_period_bars = 20 

    for thresh in THRESHOLDS:
        p_buy = probs[:, 1]
        p_sell = probs[:, 2]
        
        total_trades = 0
        buy_wins = 0
        
        i = 0
        while i < len(probs):
            # BUY LOGIC: Must beat threshold AND be higher than sell probability
            if p_buy[i] > thresh and p_buy[i] > p_sell[i]:
                # 1. Execute the trade
                total_trades += 1
                
                # 2. Check if it was a winner based on your target labels
                if y_val[i] == 1:
                    buy_wins += 1
                
                # 3. THE LOCKOUT: Fast-forward the index so we ignore all 
                # overlapping cluster signals while this trade is "open"
                i += holding_period_bars 
            else:
                # No trade entered, move to the next candle
                i += 1
        
        # Calculate Metrics
        if total_trades == 0:
            results.append({
                "Threshold": round(thresh, 2),
                "Trades": 0,
                "Win Rate": 0.0,
                "Net Score (R)": 0.0
            })
            continue
            
        # Buy Stats
        buy_losses = total_trades - buy_wins
        win_rate = buy_wins / total_trades
        
        # Net Score (1:2 R:R means a win gets +2R, a loss gets -1R)
        net_score = (buy_wins * 2) - (buy_losses * 1)
        
        results.append({
            "Threshold": round(thresh, 2),
            "Trades": total_trades,
            "Win Rate": round(win_rate * 100, 2),
            "Net Score (R)": net_score
        })

    # 3. Display Results
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv("MLP_VAE_Strategy/buy_only_btc_grid_search.csv", index=False)
    print("\nSaved full results to MLP_VAE_Strategy/buy_only_btc_grid_search.csv")

if __name__ == "__main__":
    main()