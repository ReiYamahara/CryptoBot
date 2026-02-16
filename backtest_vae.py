from VAE_mean_reversion import *
from backtesting_engine import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_path = 'vae_model.pth'
device = torch.device("cpu")

vae_model = torch.load(model_path, map_location=device)
vae_model.eval()
print("Model weights loaded successfully.")

scaler = joblib.load('vae_scaler.pkl')
device = torch.device("cpu")

strategy = SmartMeanReversion(vae_model=vae_model, scaler=scaler, device=device, risk_threshold=4.0)
engine = BackTestEngine(initial_capital=10000.0, fee=0.001)

csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('btc_data/XBTUSDT_1.csv', header=None, names=csv_headers)
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
print(df)
df = df[df['time'] >= '2025-01-01']
print(df)
print("Starting Backtest...")
engine.run(df, strategy)
print("Backtest Complete.")

results = pd.DataFrame(engine.equity_curve)
results['time'] = pd.to_datetime(results['time'])
results.set_index('time', inplace=True)

# Calculate simple metrics
total_return = (results['value'].iloc[-1] - 10000) / 10000 * 100
max_drawdown = (results['value'] / results['value'].cummax() - 1).min() * 100

print("-" * 30)
print(f"Final Value: ${results['value'].iloc[-1]:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("-" * 30)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['value'], label='Strategy Equity')
plt.title(f'VAE Strategy Backtest (Return: {total_return:.2f}%)')
plt.ylabel('Value ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

