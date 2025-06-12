# src/train.py

import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from model import make_baseline_model, SimpleHabitabilityNet

def load_data(artifact_dir, batch_size=32):
    # Load arrays
    X_train = np.load(os.path.join(artifact_dir, "X_train.npy"))
    y_train = np.load(os.path.join(artifact_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(artifact_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(artifact_dir, "y_val.npy"))

    # Baseline needs NumPy; Neural net needs tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return X_train, y_train, X_val, y_val, train_loader, val_loader

def baseline_evaluate(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred    = clf.predict(X_val)
    y_proba   = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    print("=== Baseline: Logistic Regression ===")
    print(f"Val Accuracy : {acc:.3f}")
    print(f"Val ROC-AUC : {auc:.3f}")
    print("=" * 38)

def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            total_loss += loss.item() * Xb.size(0)
            preds  = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def main():
    # Paths & settings
    artifact_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "artifacts")
    device       = torch.device("cpu")
    input_dim    = 8
    lr           = 1e-3
    epochs       = 20
    batch_size   = 32

    # Load data
    X_train, y_train, X_val, y_val, train_loader, val_loader = load_data(artifact_dir, batch_size)

    # Baseline
    baseline_evaluate(X_train, y_train, X_val, y_val)
    print("\nStarting PyTorch neural network training...\n")

    # Neural network
    model     = SimpleHabitabilityNet(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_loop(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    # Save model weights
    out_path = os.path.join(artifact_dir, "model.pth")
    torch.save(model.state_dict(), out_path)
    print(f"\nModel weights saved to {out_path}")

if __name__ == "__main__":
    main()
