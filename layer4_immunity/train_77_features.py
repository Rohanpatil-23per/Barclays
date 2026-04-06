"""
IMMUNEX - Layer 4: Train 77-Feature Model on Master Dataset
============================================================
Trains the adaptive immunity classifier on the full 77-feature CICIDS dataset.

Run:
    source ~/.venvs/immunex/bin/activate
    python layer4_immunity/train_77_features.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

# Master dataset paths
X_TRAIN_PATH = os.path.join(PROJECT_DIR, "master_dataset", "X_train.csv")
Y_TRAIN_PATH = os.path.join(PROJECT_DIR, "master_dataset", "y_train.csv")
X_TEST_PATH  = os.path.join(PROJECT_DIR, "master_dataset", "X_test.csv")
Y_TEST_PATH  = os.path.join(PROJECT_DIR, "master_dataset", "y_test.csv")
FEATURE_COLS = os.path.join(PROJECT_DIR, "master_dataset", "feature_columns.json")

MODEL_PATH  = os.path.join(MODEL_DIR, "lora_model_77f.pt")
LOG_PATH    = os.path.join(LOG_DIR, "train_77f_log.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

INPUT_DIM = 77
RANK_LEVELS = [8, 16, 32]

# ─── LoRA Layer ───────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank = rank
        self.base = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))

# ─── Model ────────────────────────────────────────────────────────────────────
class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=77, rank=8):
        super().__init__()
        self.input_dim = input_dim
        self.current_rank = rank
        # Wider architecture for 77 features
        hidden1 = 256
        hidden2 = 128
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_layer = LoRALayer(hidden2, 32, rank=rank)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))
    
    def freeze_base(self):
        for p in self.base_encoder.parameters():
            p.requires_grad = False
        for p in self.lora_layer.base.parameters():
            p.requires_grad = False
    
    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ─── Training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total * 100

def evaluate(model, X, y, device, batch_size=1024):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            preds = model(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    return accuracy_score(y, all_preds) * 100, np.array(all_preds)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("  IMMUNEX - LAYER 4: 77-FEATURE MODEL TRAINING")
    print("=" * 60)

    # ── Load feature names ────────────────────────────────────────────────────
    with open(FEATURE_COLS, 'r') as f:
        feature_names = json.load(f)
    print(f"\n  Feature columns: {len(feature_names)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading training data...")
    X_train_df = pd.read_csv(X_TRAIN_PATH)
    y_train_df = pd.read_csv(Y_TRAIN_PATH)
    X_test_df = pd.read_csv(X_TEST_PATH)
    y_test_df = pd.read_csv(Y_TEST_PATH)
    
    X_train = X_train_df.values.astype(np.float32)
    y_train = y_train_df['is_attack'].values.astype(np.int64)
    X_test = X_test_df.values.astype(np.float32)
    y_test = y_test_df['is_attack'].values.astype(np.int64)
    
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Labels (train): benign={sum(y_train==0):,}, attack={sum(y_train==1):,}")

    # ── Handle NaN/Inf ────────────────────────────────────────────────────────
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # ── Standardize features ──────────────────────────────────────────────────
    print("\n  Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    # Clip extreme values
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)

    # ── Build model ───────────────────────────────────────────────────────────
    initial_rank = RANK_LEVELS[0]
    print(f"\n  Building model (input_dim={INPUT_DIM}, rank={initial_rank})...")
    model = IMMUNEXLayer4(input_dim=INPUT_DIM, rank=initial_rank).to(device)
    criterion = nn.CrossEntropyLoss()
    
    total, trainable = model.param_counts()
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Create data loader ────────────────────────────────────────────────────
    # Use class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2
    sample_weights = class_weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(train_dataset, batch_size=512, sampler=sampler)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING (15 epochs)")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    best_acc = 0.0
    train_log = []
    t0 = time.time()
    
    for ep in range(15):
        loss, tr_acc = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        te_acc, _ = evaluate(model, X_test, y_test, device)
        
        if te_acc > best_acc:
            best_acc = te_acc
            # Save best model
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": INPUT_DIM,
                "feature_names": feature_names,
                "accuracy": best_acc,
                "lora_rank": model.current_rank,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }, MODEL_PATH)
        
        train_log.append({
            "epoch": ep + 1,
            "loss": round(loss, 4),
            "train_acc": round(tr_acc, 2),
            "test_acc": round(te_acc, 2),
        })
        print(f"  Epoch {ep+1:2d}/15 | Loss: {loss:.4f} | "
              f"Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%"
              + (" ⭐" if te_acc == best_acc else ""))
    
    elapsed = time.time() - t0
    print(f"\n✅ Training Complete!")
    print(f"   Best Test Accuracy: {best_acc:.2f}%")
    print(f"   Time: {elapsed:.1f}s")

    # ── Freeze base encoder ───────────────────────────────────────────────────
    model.freeze_base()
    _, trainable = model.param_counts()
    print(f"   🔒 Base encoder frozen")
    print(f"   Trainable now: {trainable:,} (LoRA adapters + head only)")

    # ── Classification report ─────────────────────────────────────────────────
    # Reload best model
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, pred = evaluate(model, X_test, y_test, device)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, pred, target_names=["Benign", "Attack"]))

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(LOG_PATH, "w") as f:
        json.dump({
            "training": train_log,
            "final_accuracy": best_acc,
            "input_dim": INPUT_DIM,
            "lora_rank": model.current_rank,
            "train_time_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"📋 Log saved to: {LOG_PATH}")

    # ── Update primary model path ─────────────────────────────────────────────
    # Copy to lora_model_ewc.pt (primary model used by server)
    import shutil
    primary_model = os.path.join(MODEL_DIR, "lora_model_ewc.pt")
    shutil.copy(MODEL_PATH, primary_model)
    print(f"💾 Model saved to: {primary_model}")
    print(f"   (input_dim={INPUT_DIM}, rank={model.current_rank}, acc={best_acc:.2f}%)")

    print(f"\n🎉 77-FEATURE MODEL TRAINING DONE!")
    print(f"   Start server: python layer4_immunity/server.py")

if __name__ == "__main__":
    main()
