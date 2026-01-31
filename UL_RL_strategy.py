import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm


# DATA SETUP AND CLEANING
csv_headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
df = pd.read_csv('Bitcoin Data/XBTUSDT_1.csv', header=None, names=csv_headers)

# using last 2 years of data
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
df= df[df['time'] >= '2023-01-01']
print(df.head())

# creating core features
df['log_return'] = np.log(df['close']/df['close'].shift(1))
df['sma_volume'] = df['volume'].rolling(window=60).mean()
df['relative_volume'] = df['volume']/df['sma_volume']
df['volatility'] = (df['high']-df['low'])/df['close']
def calculate_rsi(df, period=14):
    # Calculate the price change
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate Wilder's Smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['rsi'] = calculate_rsi(df)

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=8, seq_len=64):
        super(LSTM_VAE, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared encoder backbone
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True,
            bidirectional=True
        )

        # Heads for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)

        # Decoder
        self.decoder_project = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

        self.final_layer = nn.Linear(hidden_dim, input_dim)


    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n_concatenated = torch.cat((h_n[0], h_n[1]), dim=1)
        
        # 2. Get distribution parameters
        mu = self.fc_mu(h_n_concatenated)
        logvar = self.fc_logvar(h_n_concatenated)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_projected = self.decoder_project(z)
        
        # 2. "Repeat" z to create a sequence for the LSTM
        # We want input shape: (Batch, seq_len, hidden_dim)
        # So we repeat the vector 64 times
        decoder_input = z_projected.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 3. Pass through Decoder LSTM
        # out shape: (Batch, seq_len, hidden_dim)
        out, _ = self.decoder_lstm(decoder_input)
        
        # 4. Map back to features (Batch, seq_len, 4)
        x_recon = self.final_layer(out)
        
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, kl_weight=0.0001):
    """
    1. MSE Loss for reconstruction (Regression).
    2. KL Divergence for regularization.
    """
    # MSE: Compare original matrix vs reconstructed matrix
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # IMPORTANT: Financial data is noisy. 
    # If KL is too strong, the model ignores the data and outputs flat zeros (Posterior Collapse).
    # We multiply KL by a small weight to prioritize reconstruction first.
    loss = recon_loss + (kl_loss * kl_weight)
    
    return loss, recon_loss, kl_loss

class RollingWindowDataset(Dataset):
    def __init__(self, data, window_size=64):
        """
        data: Numpy array of shape (N_samples, 4_features)
        window_size: The sequence length (64)
        """
        # Convert to float32 (standard for PyTorch)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        
    def __len__(self):
        # We can create (Total - Window) sequences
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        # Slice the data from idx to idx + 64
        return self.data[idx : idx + self.window_size]


def train_vae(dataframe, window_size=64, batch_size=64, epochs=10, save_path="vae_model.pth"):
    # 1. SETUP DEVICE (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. PREPARE DATA
    # Convert DataFrame to numpy array (drop timestamp columns first!)
    data = df[['log_return', 'relative_volume', 'volatility', 'rsi']]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill(inplace=True)
    data.fillna(0, inplace=True)
    data_values = data.values
    
    dataset = RollingWindowDataset(data_values, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. INITIALIZE MODEL
    # input_dim=4 because we have 4 features
    model = LSTM_VAE(input_dim=4, hidden_dim=64, latent_dim=8, seq_len=window_size)
    model = model.to(device)

    # 4. OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. TRAINING LOOP
    for epoch in range(epochs):
        model.train() # Set mode to training (enables dropout, gradients)
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move batch to GPU
            batch = batch.to(device)
            
            # --- FORWARD PASS ---
            optimizer.zero_grad() # Reset gradients
            
            # Recon_x, Mean, LogVar, Z = model(x)
            recon_batch, mu, logvar, z = model(batch)
            
            # --- CALCULATE LOSS ---
            # Using the function we defined earlier
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch, mu, logvar)
            
            # --- BACKPROPAGATION ---
            loss.backward()  # Calculate gradients
            optimizer.step() # Update weights
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")

    # 6. SAVE THE TRAINED MODEL
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")
    return model

trained_model = train_vae(df)

def run_vae(input, batch_size):
    # data should be size (batch size, 64, 4)
    pass

