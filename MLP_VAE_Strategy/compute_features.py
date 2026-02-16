import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
from get_data import *

# Assume 'df' is your main dataframe loaded elsewhere
# df columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'symbol']

def compute_features(df_asset):
    # 1. Create a deep copy to avoid warnings
    df = df_asset.copy()
    
    # Convert Unix timestamp (seconds) to Datetime Object
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Set it as the DataFrame Index (Required for VWAP)
    df.set_index('timestamp', inplace=True)
    
    # Sort by time to be safe
    df.sort_index(inplace=True)

    # --- 1. Log Returns (Stationary Momentum) ---
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(4))
    df['log_ret_12h'] = np.log(df['close'] / df['close'].shift(12))
    df['log_ret_24h'] = np.log(df['close'] / df['close'].shift(24))

    # --- 2. Relative Strength & Oscillators ---
    # Safe RSI (FillNa prevents crashes if data is too short)
    df['rsi_14'] = ta.rsi(df['close'], length=14) / 100.0
    df['rsi_7'] = ta.rsi(df['close'], length=7) / 100.0
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None:
        df['stoch_k'] = stoch.iloc[:, 0] / 100.0 
        df['stoch_d'] = stoch.iloc[:, 1] / 100.0
    else:
        # Fallback if stoch fails
        df['stoch_k'] = 0
        df['stoch_d'] = 0

    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20) / 100.0
    df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14) / -100.0

    # --- 3. Volatility & Bands ---
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df['bb_pct'] = bbands.iloc[:, -1] 
        df['bb_width'] = bbands.iloc[:, 3] 
    
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
    df['std_20'] = df['close'].rolling(20).std() / df['close']
    df['donchian_width'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    
    # Return/Vol Ratio
    df['return_vol_ratio'] = df['log_ret_1h'].rolling(24).mean() / (df['log_ret_1h'].rolling(24).std() + 1e-9)

    # --- 4. Trend Strength ---
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        df['adx'] = adx_df.iloc[:, 0] / 100.0
    
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd_line'] = macd.iloc[:, 0] / df['close']
        df['macd_hist'] = macd.iloc[:, 1] / df['close']
    
    aroon = ta.aroon(df['high'], df['low'], length=25)
    if aroon is not None:
        df['aroon_up'] = aroon.iloc[:, 0] / 100.0
        df['aroon_down'] = aroon.iloc[:, 1] / 100.0
        df['aroon_osc'] = aroon.iloc[:, 2] / 100.0

    # --- 5. Moving Average Distances ---
    for length in [7, 25, 99, 200]:
        sma = ta.sma(df['close'], length=length)
        ema = ta.ema(df['close'], length=length)
        # Check if MA exists before dividing
        if sma is not None:
            df[f'dist_sma_{length}'] = (df['close'] / sma) - 1
        if ema is not None:
            df[f'dist_ema_{length}'] = (df['close'] / ema) - 1

    # --- 6. Volume & Flow ---
    df['vol_change_1'] = np.log(df['volume'] / df['volume'].shift(1).replace(0, 1))
    df['rel_vol_24'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-9)
    
    obv = ta.obv(df['close'], df['volume'])
    if obv is not None:
        df['obv_norm'] = obv / (df['volume'].rolling(200).sum() + 1e-9)
    
    df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14) / 100.0
    
    # --- VWAP FIX ---
    # Now that Index is Datetime, this will work.
    vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    if vwap is not None:
        df['dist_vwap'] = (df['close'] / vwap) - 1
    else:
        df['dist_vwap'] = 0 # Fallback

    eom = ta.eom(df['high'], df['low'], df['close'], df['volume'], length=14)
    if eom is not None:
        df['eom_squashed'] = np.tanh(eom / 1e6)
    
    # Drop NaNs
    df.dropna(inplace=True)
    
    # Reset index so 'timestamp' becomes a column again
    df.reset_index(inplace=True)
    
    return df

if __name__ == "__main__":
    df = big_df.copy()
    assets = df['symbol'].unique()
    processed_dfs = []

    print(f"Processing {len(assets)} assets...")

    for asset in tqdm(assets):
        asset_data = df[df['symbol'] == asset]
        try:
            # Skip assets that are too short for calculation (need at least 200 rows)
            if len(asset_data) < 250:
                continue
                
            processed_data = compute_features(asset_data)
            processed_dfs.append(processed_data)
        except Exception as e:
            print(f"Error processing {asset}: {e}")

    final_df = pd.concat(processed_dfs)

    # Keep specific columns
    cols_to_keep = ['timestamp', 'symbol', 'close', 'high', 'low'] + \
                [c for c in final_df.columns if c not in ['open', 'high', 'low', 'volume', 'trades', 'timestamp', 'symbol', 'close']]
    final_df = final_df[cols_to_keep]
    final_df.to_csv('MLP_VAE_Strategy/train_val_test_datasets/features_dataset.csv')


    print("Processing Complete!")