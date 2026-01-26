import requests
import pandas as pd
import time
from datetime import datetime, timedelta

csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('master_q4/XBTUSDT_5.csv', header=None, names=csv_headers)
print(df.head())
print(df.columns)
print(len(df))

