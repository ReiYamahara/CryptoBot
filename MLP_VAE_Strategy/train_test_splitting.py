import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. LOAD AND PREP
print("Loading dataset...")
df = pd.read_csv("MLP_VAE_Strategy/features_dataset.csv")

# Force UTC and Sort
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Total Data Rows: {len(df)}")

# ---------------------------------------------------------
# 3. DEFINE SPLIT DATES
# ---------------------------------------------------------
VAL_START_DATE = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START_DATE = pd.Timestamp("2025-07-01", tz="UTC")

# Chronological Split
train = df[df['timestamp'] < VAL_START_DATE].copy()
val   = df[(df['timestamp'] >= VAL_START_DATE) & (df['timestamp'] < TEST_START_DATE)].copy()
test  = df[df['timestamp'] >= TEST_START_DATE].copy()

# ---------------------------------------------------------
# 4. SCALING & FEATURE MANAGEMENT
# ---------------------------------------------------------
# Define the features properly. Exclude non-math columns.
non_feature_cols = ['timestamp', 'symbol', 'label', 'open', 'high', 'low', 'close', 'volume', 'trades', 'Unnamed: 0']
feature_cols = [c for c in train.columns if c not in non_feature_cols]

print(f"\nFeature Count: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# SAVE THE COLUMN LIST (Fixes the 'Shuffle' Bug)
joblib.dump(feature_cols, "MLP_VAE_Strategy/feature_columns.pkl")
print("-> Saved feature_columns.pkl (CRITICAL for live alignment)")

# Initialize and Fit Scaler (ONLY on Training Data)
scaler = StandardScaler()
scaler.fit(train[feature_cols])

# SAVE THE SCALER (Required for Live/Backtest)
joblib.dump(scaler, "MLP_VAE_Strategy/std_scaler.pkl")
print("-> Saved std_scaler.pkl")

# Transform all sets
train[feature_cols] = scaler.transform(train[feature_cols])
val[feature_cols]   = scaler.transform(val[feature_cols])
test[feature_cols]  = scaler.transform(test[feature_cols])

# ---------------------------------------------------------
# 5. SAVE SPLITS
# ---------------------------------------------------------
print("\nSAVING DATASETS...")
train.to_csv("MLP_VAE_Strategy/train_data.csv", index=False)
val.to_csv("MLP_VAE_Strategy/val_data.csv", index=False)
test.to_csv("MLP_VAE_Strategy/test_data.csv", index=False)

print(f"1. Training Set (2021-2024):  {len(train)} rows")
print(f"2. Validation Set (H1 2025):  {len(val)} rows")
print(f"3. Test Set (H2 2025):        {len(test)} rows")
print("\nDone. Ready for Training.")