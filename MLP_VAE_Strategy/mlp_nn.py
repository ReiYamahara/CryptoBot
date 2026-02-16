import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import os
import joblib
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
BATCH_SIZE = 32       
LEARNING_RATE = 0.001 
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- INFERENCE THRESHOLD ---
CONFIDENCE_THRESHOLD = 0.5 

# ---------------------------------------------------------
# 1. LOAD DATA (STRICT FEATURE ENFORCEMENT)
# ---------------------------------------------------------
def load_labeled_data(filepath, feature_cols):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Strictly extract only the approved features (Prevents vae_error leakage)
    features = df[feature_cols].values
    
    # 2. Data is ALREADY SCALED from the earlier split pipeline. 
    # Do NOT apply StandardScaler here.
    
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(df['mlp_label'].values, dtype=torch.long)
    
    return X, y

# ---------------------------------------------------------
# 2. PHYSICAL UNDERSAMPLING
# ---------------------------------------------------------
def undersample_majority_class(X, y):
    print("\n--- UNDERSAMPLING MAJORITY CLASS ---")
    if isinstance(X, torch.Tensor): X = X.numpy()
    if isinstance(y, torch.Tensor): y = y.numpy()
        
    idx_hold = np.where(y == 0)[0]
    idx_buy = np.where(y == 1)[0]
    idx_sell = np.where(y == 2)[0]
    
    print(f"Original Counts: Hold={len(idx_hold)}, Buy={len(idx_buy)}, Sell={len(idx_sell)}")
    
    # Balance by averaging the minority classes
    n_target = int((len(idx_buy) + len(idx_sell)) / 2) 
    n_target = min(n_target, len(idx_hold))

    np.random.seed(42)
    idx_hold_downsampled = np.random.choice(idx_hold, size=n_target, replace=False)
        
    final_indices = np.concatenate([idx_hold_downsampled, idx_buy, idx_sell])
    np.random.shuffle(final_indices) 
    
    X_balanced = torch.tensor(X[final_indices], dtype=torch.float32)
    y_balanced = torch.tensor(y[final_indices], dtype=torch.long)
    
    unique, counts = np.unique(y_balanced.numpy(), return_counts=True)
    print(f"New Counts: {dict(zip(unique, counts))}")
    
    return X_balanced, y_balanced

# ---------------------------------------------------------
# 3. MLP ARCHITECTURE
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------
def main():
    # 1. Load the universal feature map
    feature_cols = joblib.load("MLP_VAE_Strategy/feature_columns.pkl")
    input_dim = len(feature_cols)
    print(f"Loaded Feature Map: {input_dim} columns expected.")

    # 2. Load Data
    X_train, y_train = load_labeled_data("MLP_VAE_Strategy/train_labeled.csv", feature_cols)
    X_val, y_val = load_labeled_data("MLP_VAE_Strategy/val_labeled.csv", feature_cols)
    
    # 3. Apply Undersampling (Train Only)
    # X_train_bal, y_train_bal = undersample_majority_class(X_train, y_train)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    model = TradingMLP(input_dim).to(DEVICE)

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train.numpy()), 
        y=y_train.numpy()
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class Weights applied: Hold={weights_tensor[0]:.4f}, Buy={weights_tensor[1]:.4f}, Sell={weights_tensor[2]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25)
    
    best_val_f1 = 0 
    patience = 100
    trigger_times = 0
    
    print("\nStarting Training (Class-weight balanced data)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(DEVICE))
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.numpy())
        
        # MACRO F1 is better here. Weighted F1 will be artificially high because 
        # the validation set is 90% "Hold", hiding poor Buy/Sell performance.
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} | Val F1 (Macro): {val_f1:.4f}")

        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            trigger_times = 0
            torch.save(model.state_dict(), "MLP_VAE_Strategy/mlp_model_best.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stopping Triggered!")
                break
                
    # ---------------------------------------------------------
    # 5. FINAL REPORT WITH THRESHOLDS
    # ---------------------------------------------------------
    print(f"\nLoading Best Model for Threshold Eval (Threshold={CONFIDENCE_THRESHOLD})...")
    model.load_state_dict(torch.load("MLP_VAE_Strategy/mlp_model_best.pth", weights_only=True))
    model.eval()
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(DEVICE))
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(probs)):
                p_hold, p_buy, p_sell = probs[i]
                
                # Exact logic from our Grid Search
                if p_buy > CONFIDENCE_THRESHOLD and p_buy >= p_sell:
                    all_preds.append(1)
                elif p_sell > CONFIDENCE_THRESHOLD and p_sell > p_buy:
                    all_preds.append(2)
                else:
                    all_preds.append(0)
            
            all_targets.extend(labels.numpy())
            
    print("\n--- CLASSIFICATION REPORT (With Thresholds) ---")
    # zero_division=0 prevents crashes if a class isn't predicted
    print(classification_report(all_targets, all_preds, target_names=['Hold', 'Buy', 'Sell'], zero_division=0))
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Hold', 'Buy', 'Sell'], 
                yticklabels=['Hold', 'Buy', 'Sell'])
    plt.title(f"Confusion Matrix (Threshold > {CONFIDENCE_THRESHOLD})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("MLP_VAE_Strategy/confusion_matrix.png")
    print("Confusion matrix saved to MLP_VAE_Strategy/confusion_matrix.png")

if __name__ == "__main__":
    main()