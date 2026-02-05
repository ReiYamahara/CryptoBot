import pandas as pd
import numpy as np

# CHECKS THE DATA BEING GENERATED FROM ORDER_BOOK.PY
# data = pd.read_parquet("kraken_data/BTC_USD_1769551499_0.parquet")
# print(data)

csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('Bitcoin Data/XBTUSDT_1.csv', header=None, names=csv_headers)

# using last 2 years of data
df['time'] = pd.to_datetime(df['timestamp'], unit='s')


df['log_return'] = np.log(df['close']/df['close'].shift(1))
df['sma_volume'] = df['volume'].rolling(window=64).mean()
df['relative_volume'] = df['volume']/df['sma_volume']
df['volatility'] = (df['high']-df['low'])/df['close']

print(df['time'].head())