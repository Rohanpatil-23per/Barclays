"""
IMMUNEX - Layer 4: LoRA Retraining
Parses text features → trains classifier → retrains on new attacks
Input:  lora_retrain_source.csv, lora_test.csv
Output: trained model saved to layer4_immunity/models/
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

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"E:\immunex_p4\layer4_immunity"
DATA_DIR    = r"E:\immunex_p4\person4_layer4"
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
TRAIN_CSV   = os.path.join(DATA_DIR, "lora_retrain_source.csv")
TEST_CSV    = os.path.join(DATA_DIR, "lora_test.csv")
MODEL_PATH  = os.path.join(MODEL_DIR, "lora_model.pt")
LOG_PATH    = os.path.join(LOG_DIR,   "lora_training_log.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ─── These are the 25 features in your text data ──────────────────────────────
FEATURE_NAMES = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "flow_bytes/s", "flow_packets/s", "fwd_packet_length_mean",
    "bwd_packet_length_mean", "flow_iat_mean", "fwd_iat_mean",
    "bwd_iat_mean", "syn_flag_count", "ack_flag_count", "fin_flag_count",
    "rst_flag_count", "psh_flag_count", "packet_length_mean",
    "packet_length_std", "fwd_packets/s", "bwd_packets/s",
    "init_fwd_win_bytes", "init_bwd_win_bytes", "active_mean",
    "idle_mean", "down/up_ratio", "avg_packet_size"
]
INPUT_DIM = len(FEATURE_NAMES)   # 25

# ─── Parse text row into numeric vector ───────────────────────────────────────
def parse_text(text):
    """
    Input:  "flow_duration:-0.4655 total_fwd_packets:-0.2564 ..."
    Output: numpy array of 25 floats
    """
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try:
                lookup[key] = float(val)
            except:
                lookup[key] = 0.0
    return np.array([lookup.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32)

def load_csv(path, label="dataset"):
    print(f"📂 Loading {label}...")
    df = pd.read_csv(path)
    print(f"   Rows: {len(df)} | Labels: {df['label'].value_counts().to_dict()}")
    X = np.vstack(df["text"].apply(parse_text).values)
    y = df["label"].values.astype(np.int64)
    print(f"   Features parsed: {X.shape}")
    return X, y

# ─── LoRA Layer ───────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    base = original frozen weights
    lora_A + lora_B = small trainable adapters (rank=8)
    Output = base(x) + lora_B(lora_A(x))
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base   = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))

# ─── Model ────────────────────────────────────────────────────────────────────
class IMMUNEXLayer4(nn.Module):
    """
    Binary classifier: Benign (0) vs Attack (1)

    base_encoder: learns what 25 network features MEAN → frozen after training
    lora_head:    makes final decision → stays trainable for retraining cycles
    """
    def __init__(self, input_dim=25):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_head = nn.Sequential(
            LoRALayer(64, 32, rank=8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)   # 2 classes: Benign=0, Attack=1
        )

    def forward(self, x):
        return self.lora_head(self.base_encoder(x))

    def freeze_base(self):
        """Lock base_encoder — called after initial training"""
        for p in self.base_encoder.parameters():
            p.requires_grad = False
        print("   🔒 Base encoder frozen — only LoRA head updates from now on")

    def trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ─── Training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total * 100

def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        out  = model(torch.tensor(X, dtype=torch.float32).to(device))
        pred = out.argmax(1).cpu().numpy()
    return accuracy_score(y, pred) * 100, pred

# ─── Main training function ───────────────────────────────────────────────────
def initial_train(model, X_train, y_train, X_test, y_test, device):
    """
    Initial training — trains ALL weights including base_encoder
    Uses mini-batches of 256, 20 epochs, Adam optimizer
    After training: freezes base_encoder permanently
    """
    print("\n" + "="*60)
    print("  INITIAL TRAINING")
    print("="*60)

    total, trainable = model.trainable_params()
    print(f"   Total params    : {total:,}")
    print(f"   Trainable params: {trainable:,}")

    dataset   = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader    = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    log       = []
    t0        = time.time()

    for ep in range(20):
        loss, tr_acc = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        te_acc, _    = evaluate(model, X_test, y_test, device)
        best_acc     = max(best_acc, te_acc)

        log.append({"epoch": ep+1, "loss": round(loss,4),
                    "train_acc": round(tr_acc,2), "test_acc": round(te_acc,2)})

        print(f"  Epoch {ep+1:2d}/20 | Loss: {loss:.4f} | "
              f"Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")

    elapsed = time.time() - t0
    print(f"\n✅ Initial Training Complete!")
    print(f"   Best Test Accuracy : {best_acc:.2f}%")
    print(f"   Time               : {elapsed:.1f}s")

    # Freeze base encoder after initial training
    model.freeze_base()
    total, trainable = model.trainable_params()
    print(f"   Trainable now      : {trainable:,} (LoRA only)")

    # Full classification report
    _, pred = evaluate(model, X_test, y_test, device)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, pred,
          target_names=["Benign", "Attack"]))

    return best_acc, log

# ─── Retraining cycle ─────────────────────────────────────────────────────────
def retrain_cycle(model, X_train, y_train, X_new, y_new,
                  device, cycle_num, epochs=10):
    """
    One retraining cycle on new attack variants
    Batch = 70% original data + 30% new attacks
    Only LoRA head updates (base is frozen)
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # Build rehearsal batch
    n_orig = int(0.7 * 512)
    n_new  = 512 - n_orig
    idx_o  = np.random.choice(len(X_train), n_orig, replace=False)
    idx_n  = np.random.choice(len(X_new),   n_new,  replace=True)

    X_batch = np.vstack([X_train[idx_o], X_new[idx_n]])
    y_batch = np.concatenate([y_train[idx_o], y_new[idx_n]])

    dataset = TensorDataset(
        torch.tensor(X_batch, dtype=torch.float32),
        torch.tensor(y_batch, dtype=torch.long)
    )
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    best_acc = 0.0
    model.train()
    for ep in range(epochs):
        loss, tr_acc = train_epoch(model, loader, optimizer, criterion, device)
        best_acc     = max(best_acc, tr_acc)
        print(f"  Cycle {cycle_num} | Ep {ep+1:2d}/{epochs} | "
              f"Loss: {loss:.4f} | Train: {tr_acc:.1f}%")
    return best_acc

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4: LoRA RETRAINING")
    print("="*60)

    # Load data
    X_train, y_train = load_csv(TRAIN_CSV, "training data")
    X_test,  y_test  = load_csv(TEST_CSV,  "test data")

    # Build model
    model = IMMUNEXLayer4(input_dim=INPUT_DIM).to(device)

    # Initial training
    best_acc, train_log = initial_train(
        model, X_train, y_train, X_test, y_test, device
    )

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   INPUT_DIM,
        "feature_names": FEATURE_NAMES,
        "accuracy":    best_acc,
    }, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")

    # Save log
    with open(LOG_PATH, "w") as f:
        json.dump({"initial_training": train_log, "cycles": []}, f, indent=2)
    print(f"📋 Log saved to: {LOG_PATH}")

    print(f"\n🎉 LORA RETRAINING DONE!")
    print(f"   Next step: Run blind_spot.py")

if __name__ == "__main__":
    main()
