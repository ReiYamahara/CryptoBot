from pathlib import Path
import pandas as pd

# folder containing your 52 csv files
DATA_DIR = Path("MLP_VAE_Strategy/token_data")   # change if needed

cols = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

dfs = []
for fp in sorted(DATA_DIR.glob("*.csv")):
    symbol = fp.stem.split("_")[0]  # "XBTUSDT_60" -> "XBTUSDT"

    df = pd.read_csv(fp, header=None, names=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # If your timestamps are already ISO strings instead of unix seconds, use:
    # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["symbol"] = symbol
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)

# optional: sort within symbol by time
big_df = big_df.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)

print(big_df.shape)
print(big_df.head())
