from MLP_VAE_Strategy.MLP_VAE_Strategy import MLPStrategy
from backtesting_engine import BackTestEngine 
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# --- CONFIG ---
MLP_PATH = "MLP_VAE_Strategy/mlp_model_best.pth"
VAE_PATH = "MLP_VAE_Strategy/vae_model_mlp.pth"

# NOTE: Scalers are now loaded internally by the Strategy class
# from "MLP_VAE_Strategy/tech_scaler.pkl" and "vae_scaler.pkl"

# Match this to your winning config
WINNING_VAE_CONFIG = {
    'hidden_1': 64,
    'hidden_2': 32,
    'latent_dim': 6
}

def load_data(filepath):
    print(f"Loading {filepath}...")
    csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
    df = pd.read_csv(filepath, header=None, names=csv_headers)
    
    # Ensure timestamp is sorted and index is datetime for easy slicing
    df['time'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    # Check if scalers exist before running
    if not os.path.exists("MLP_VAE_Strategy/std_scaler.pkl"):
        raise FileNotFoundError("Run the split_scaler script first! Missing tech_scaler.pkl")

    device = torch.device("cpu") # CPU is faster for sequential backtesting
    
    # 1. LOAD DATA
    # Point this to your actual 1-minute data file
    df = load_data('btc_data/XBTUSDT_60.csv')

    # 2. SEPARATE WARMUP DATA vs TEST DATA
    start_date = '2025-01-01'
    end_date = '2025-07-01'
    
    # TEST DATA: The period we want to simulate trades on
    test_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    test_df.reset_index(inplace=True) # Engine usually iterates rows, needs index as column sometimes

    # WARMUP DATA: The 2000 bars BEFORE the start date
    # This ensures indicators (EMA, RSI) are ready on Minute 1 of the test.
    warmup_df = df[df.index < start_date].tail(500).copy()
    
    print(f"Test Data: {len(test_df)} rows (Starts {test_df['time'].iloc[0]})")
    print(f"Warmup Data: {len(warmup_df)} rows")

    # 3. INITIALIZE STRATEGY
    # REMOVED: scaler_path and training_data_path (Handled internally now)
    strategy = MLPStrategy(
        mlp_path=MLP_PATH,
        vae_path=VAE_PATH,
        vae_config=WINNING_VAE_CONFIG,
        threshold=0.44,
        device=device
    )

    # --- INJECT WARMUP DATA (Optimized) ---
    print("Injecting Warmup Data...")
    
    # We reset index to ensure 'timestamp' is a column, not the index
    warmup_df.reset_index(drop=True, inplace=True)
    
    # Bulk load is faster than looping
    strategy.buffer = warmup_df.copy()
    
    print(f"Strategy Buffer Pre-filled with {len(strategy.buffer)} bars.")

    # 4. RUN BACKTEST
    # Initialize Engine (Ensure your engine class exists and accepts these args)
    engine = BackTestEngine(initial_capital=10000.0, fee=0.000) # Binance taker fee approx 0.06%
    
    print("\n--- STARTING BACKTEST ---")
    engine.run(test_df, strategy)
    print("--- BACKTEST COMPLETE ---")

    # 5. RESULTS
    if len(engine.equity_curve) > 0:
        results = pd.DataFrame(engine.equity_curve)
        results['time'] = pd.to_datetime(results['time'])
        results.set_index('time', inplace=True)

        final_val = results['value'].iloc[-1]
        total_return = (final_val - 10000) / 10000 * 100
        
        print(f"Final Capital: ${final_val:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['value'], label='MLP Equity', color='blue')
        
        # Benchmark (Buy & Hold)
        if len(test_df) > 0:
            benchmark = (test_df['close'] / test_df['close'].iloc[0]) * 10000
            # Align timestamps
            plt.plot(test_df['time'], benchmark, label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
        
        plt.title(f"MLP Strategy (Split Scaler Fix): {total_return:.2f}% Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No trades were made or equity curve is empty.")