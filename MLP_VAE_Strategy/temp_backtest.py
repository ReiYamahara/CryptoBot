import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

# Make sure this matches your exact architecture
import torch.nn as nn
class TradingMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TradingMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, num_classes)
        self.leaky_relu = nn.LeakyReLU() 

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.leaky_relu(self.bn3(self.layer3(x)))
        return self.output(x)

def load_validation_data(filepath, feature_cols):
    print(f"Loading {filepath} for Grid Search...")
    df = pd.read_csv(filepath)
    features = df[feature_cols].values
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(df['mlp_label'].values, dtype=torch.long)
    return X, y

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Data & Model
    feature_cols = joblib.load("MLP_VAE_Strategy/feature_columns.pkl")
    input_dim = len(feature_cols)
    X_val, y_val = load_validation_data("MLP_VAE_Strategy/val_labeled.csv", feature_cols)
    
    model = TradingMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load("MLP_VAE_Strategy/mlp_model_best.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Get Raw Probabilities
    print("Running validation data through model...")
    with torch.no_grad():
        inputs = X_val.to(DEVICE)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
    targets = y_val.numpy()
    
    # 3. Test Thresholds from 0.34 to 0.85
    # (0.34 is just above the 0.33 random guessing baseline for 3 classes)
    thresholds = np.arange(0.34, 0.86, 0.02)
    
    print("\n--- CONFIDENCE THRESHOLD GRID SEARCH ---")
    print(f"{'Threshold':<10} | {'Total Buys':<12} | {'Buy Precision':<15} | {'Buy Recall':<12} | {'Macro F1':<10}")
    print("-" * 75)
    
    for t in thresholds:
        preds = []
        for i in range(len(probs)):
            p_hold, p_buy, p_sell = probs[i]
            
            # If the highest probability beats the threshold, take the trade
            if p_buy > t and p_buy >= p_sell:
                preds.append(1)
            elif p_sell > t and p_sell > p_buy:
                preds.append(2)
            else:
                preds.append(0)
                
        # Calculate Metrics (zero_division=0 prevents crashes when 0 trades trigger)
        precisions = precision_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
        recalls = recall_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        
        buy_prec = precisions[1] * 100
        buy_rec = recalls[1] * 100
        total_buys = sum(1 for p in preds if p == 1)
        
        print(f"{t:.2f}       | {total_buys:<12} | {buy_prec:>5.2f}%          | {buy_rec:>5.2f}%       | {macro_f1:.4f}")

if __name__ == "__main__":
    main()