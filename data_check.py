import pandas as pd

# CHECKS THE DATA BEING GENERATED FROM ORDER_BOOK.PY
data = pd.read_parquet("kraken_data/BTC_USD_1769551499_0.parquet")
print(data)
