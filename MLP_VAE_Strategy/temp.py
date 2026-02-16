import pandas as pd
import numpy as np

# Load your labeled training file
df = pd.read_csv("MLP_VAE_Strategy/train_labeled.csv")

print(np.max(df['cci']), np.min(df['cci']))
