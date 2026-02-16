import pandas as pd
import joblib
from MLP_VAE_Strategy.MLP_VAE_Strategy import MLPStrategy

# --- CONFIG ---
# We need to verify these files match
SCALER_COLS_PATH = "MLP_VAE_Strategy/feature_columns.pkl"
SAMPLE_DATA = "btc_data/XBTUSDT_60.csv" 

def main():
    print("--- DIAGNOSTIC: FEATURE ALIGNMENT ---")
    
    # 1. Load what the Scaler/Model expects
    expected_cols = joblib.load(SCALER_COLS_PATH)
    print(f"1. Model Expects: {len(expected_cols)} columns")
    print(f"   First 5: {expected_cols[:5]}")
    
    # 2. Simulate what the Strategy generates
    # We instantiate the class just to access 'derive_features'
    # (Args don't matter here, we just need the method)
    dummy_strategy = MLPStrategy(
        mlp_path="MLP_VAE_Strategy/mlp_model_weighted.pth", # distinct dummy paths not needed if code handles it, 
        vae_path="MLP_VAE_Strategy/vae_model_mlp.pth",      # but keeping simple
        vae_config={'hidden_1':48, 'hidden_2':32, 'latent_dim':4}
    )
    
    # Load a tiny chunk of data
    df = pd.read_csv(SAMPLE_DATA, header=None, names=['timestamp','open','high','low','close','volume','trades'], nrows=500)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Generate features exactly like the live bot does
    print("\n2. Generating Strategy Features...")
    generated_df = dummy_strategy.derive_features(df)
    
    # Extract the columns the strategy *would* select
    # (Using the same logic as on_data: filtering by expected_cols)
    available_cols = [c for c in expected_cols if c in generated_df.columns]
    
    print(f"   Strategy Generated: {len(generated_df.columns)} total features")
    print(f"   Matching Features:  {len(available_cols)} (Should be {len(expected_cols)})")
    
    # 3. CRITICAL COMPARISON
    print("\n--- MISMATCH CHECK ---")
    
    # Check 1: Are any expected columns MISSING?
    missing = set(expected_cols) - set(generated_df.columns)
    if missing:
        print(f"❌ CRITICAL: Strategy is FAILING to generate these columns:\n   {missing}")
    else:
        print("✅ All expected columns are present.")
        
    # Check 2: Is the ORDER correct?
    # The scaler transforms based on POSITION, not name.
    # If the strategy produces dataframe columns in a different order than the scaler was fit on... disaster.
    
    # The strategy calculates *many* columns, but usually selects them by name.
    # Let's verify the values of the first row to see if they make sense.
    
    first_row = generated_df.iloc[-1]
    print("\n--- VALUE SANITY CHECK (Latest Bar) ---")
    for col in expected_cols[:37]:
        val = first_row[col]
        print(f"   {col}: {val:.6f}")
        
    print("\n👉 IF these values look wrong (e.g. 'log_ret' is 95000), your column map is broken.")

if __name__ == "__main__":
    main()