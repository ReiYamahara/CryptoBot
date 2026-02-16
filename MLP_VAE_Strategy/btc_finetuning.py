import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import joblib

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.46  # Entry Trigger
TARGET_SYMBOL = "XBT"        # Bitmex/Bitcoin Symbol

# --- STEP 2 CONFIG: THE MANAGER ---
TAKE_PROFIT_PCT = 0.04       # Target 4% gain (Swing Trade)
STOP_LOSS_PCT   = 0.02       # Risk 2% loss
TRADING_FEE     = 0.001      # 0.1% Fee
INITIAL_CAPITAL = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 1024 

# --- MODEL ARCHITECTURE ---
class TradingMLP(nn.Module):
    def __init__(self, input_dim=36, num_classes=3):
        super(TradingMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.leaky_relu = nn.LeakyReLU() 
    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        x = self.leaky_relu(self.layer3(x))
        return self.output(x)

def run_manager_backtest():
    print("--- LOADING DATA ---")
    df = pd.read_csv("MLP_VAE_Strategy/val_labeled.csv")
    
    # 1. FILTER FOR BITCOIN
    mask = df['symbol'].str.contains(TARGET_SYMBOL, case=False, na=False)
    btc_df = df[mask].copy()
    
    if len(btc_df) == 0:
        print(f"ERROR: No symbol found matching '{TARGET_SYMBOL}'")
        return

    # 2. PREPARE DATA (Keep Prices for TP/SL simulation)
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
    
    # We need Open/High/Low/Close for realistic simulation
    # If High/Low are missing, we approximate with Close (Less accurate but functional)
    prices_open = btc_df['open'].values if 'open' in btc_df.columns else btc_df['close'].values
    prices_high = btc_df['high'].values if 'high' in btc_df.columns else btc_df['close'].values
    prices_low  = btc_df['low'].values  if 'low' in btc_df.columns else btc_df['close'].values
    prices_close= btc_df['close'].values
    timestamps  = pd.to_datetime(btc_df['timestamp'])
    
    # Features
    cols_to_drop = ['timestamp', 'symbol', 'label', 'open', 'high', 'low', 'close', 
                    'volume', 'trades', 'Unnamed: 0', 'future_return', 
                    'mlp_label', 'vae_threshold', 'future_log_return', 'paper_label']
    feature_cols = [c for c in btc_df.columns if c not in cols_to_drop]
    features = btc_df[feature_cols].values
    
    # Scale
    scaler = joblib.load("MLP_VAE_Strategy/scaler.pkl")
    features = scaler.transform(features)
    
    X = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. INFERENCE (GET ENTRY SIGNALS)
    model = TradingMLP(input_dim=len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load("MLP_VAE_Strategy/mlp_model_weighted.pth", map_location=DEVICE))
    model.eval()
    
    print("Generating Entry Signals...")
    all_probs = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].to(DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    # Buy probability > Threshold = ENTRY SIGNAL
    entry_signals = (all_probs[:, 1] > CONFIDENCE_THRESHOLD).astype(int)
    
    # --- 4. THE MANAGER SIMULATION ---
    print("Running Manager Logic (TP/SL)...")
    
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    
    in_position = False
    entry_price = 0
    hold_duration = 0
    
    trades = []
    
    for i in range(len(prices_close) - 1):
        current_close = prices_close[i]
        current_high  = prices_high[i]
        current_low   = prices_low[i]
        
        # --- LOGIC IF WE ARE HOLDING ---
        if in_position:
            hold_duration += 1
            
            # Check Stop Loss (Hit Low?)
            sl_price = entry_price * (1 - STOP_LOSS_PCT)
            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            
            # Did we hit SL?
            if current_low <= sl_price:
                # Sold at SL
                exit_price = sl_price
                pnl = (exit_price - entry_price) / entry_price
                pnl -= (TRADING_FEE * 2) # Entry + Exit fee
                
                capital *= (1 + pnl)
                in_position = False
                trades.append({'type': 'SL', 'pnl': pnl, 'duration': hold_duration})
            
            # Did we hit TP?
            elif current_high >= tp_price:
                # Sold at TP
                exit_price = tp_price
                pnl = (exit_price - entry_price) / entry_price
                pnl -= (TRADING_FEE * 2)
                
                capital *= (1 + pnl)
                in_position = False
                trades.append({'type': 'TP', 'pnl': pnl, 'duration': hold_duration})
                
            # (Optional) Time limit? 
            # elif hold_duration > 100: ... force close ...
            
        # --- LOGIC IF WE ARE FLAT ---
        else:
            # Check for Model Entry Signal
            if entry_signals[i] == 1:
                in_position = True
                entry_price = current_close # Assume we buy at close
                hold_duration = 0
                
        equity_curve.append(capital)

    # Convert to arrays for plotting
    equity_curve = np.array(equity_curve)
    
    # Stats
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    if len(trades) > 0:
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        avg_dur = np.mean([t['duration'] for t in trades])
    else:
        win_rate = 0
        avg_dur = 0
        
    print(f"\n--- MANAGER RESULTS ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate:     {win_rate:.1f}%")
    print(f"Avg Duration: {avg_dur:.1f} candles")
    print(f"Final Capital:${capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equity_curve, label='Manager Strategy (TP/SL)', color='blue')
    # Compare with Market
    market_curve = (prices_close / prices_close[0]) * INITIAL_CAPITAL
    plt.plot(timestamps, market_curve, label='Buy & Hold', color='gray', alpha=0.3)
    
    plt.title(f"Strategy with TP={TAKE_PROFIT_PCT*100}% SL={STOP_LOSS_PCT*100}%")
    plt.legend()
    plt.savefig("btc_manager_backtest.png")
    print("Saved chart.")

if __name__ == "__main__":
    run_manager_backtest()