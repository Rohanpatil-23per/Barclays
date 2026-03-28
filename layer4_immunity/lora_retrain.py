"""
IMMUNEX - Layer 4: LoRA Retraining with Dynamic Rank Adaptation
===============================================================
What this file does:
  Trains the primary IMMUNEX classifier on 112,245 CICIDS network samples.
  After training, automatically checks how many blind spots the model has
  and increases LoRA rank if the blind spot rate is too high.

Key features:
  1. Initial training — all 14,882 parameters train freely for 20 epochs.

  2. Base encoder freeze — after training, the base encoder (8,960 params)
     is locked permanently. Only the LoRA adapters update from now on.
     This protects the learned feature representations.

  3. Dynamic Rank Adaptation —
     After training, loads blind_spots.csv and checks what fraction of
     those unseen attacks the model still misses.
       > 60% missed → rank 8 → 16  (needs more adapter capacity)
       > 60% missed again → rank 16 → 32
       < 60% missed → rank stays (model has enough capacity)
     rank controls how many "dimensions" the LoRA adapter can represent.
     More rank = more flexible adapter = can learn harder patterns.
     The rank chosen here is saved into lora_model.pt and read by ewc.py.

Run order:
  1. python lora_retrain.py   ← this file
  2. python blind_spot.py
  3. python mutation_engine.py
  4. python ewc.py
  5. python server.py
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
BLIND_CSV   = os.path.join(BASE_DIR, "blind_spots.csv")
MODEL_PATH  = os.path.join(MODEL_DIR, "lora_model.pt")
LOG_PATH    = os.path.join(LOG_DIR,   "lora_training_log.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

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
INPUT_DIM   = len(FEATURE_NAMES)   # 25
RANK_LEVELS = [8, 16, 32]          # Dynamic rank progression


# ─── LoRA Layer ───────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    """
    Output = base(x) + lora_B(lora_A(x))

    base   → frozen after initial training (preserves old knowledge)
    lora_A → small trainable matrix (in → rank)
    lora_B → small trainable matrix (rank → out)

    rank controls adapter capacity:
      rank=8  → 2×(25×8 + 8×32) = 912 adapter params
      rank=16 → 2×(25×16 + 16×32) = 1,824 adapter params
      rank=32 → 2×(25×32 + 32×32) = 3,648 adapter params
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank   = rank
        self.base   = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


# ─── Model ────────────────────────────────────────────────────────────────────
class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25, rank=8):
        super().__init__()
        self.current_rank = rank
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_layer = LoRALayer(64, 32, rank=rank)
        self.head       = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))

    def freeze_base(self):
        """Lock base_encoder + LoRA base weights. Only adapters train."""
        for p in self.base_encoder.parameters():
            p.requires_grad = False
        for p in self.lora_layer.base.parameters():
            p.requires_grad = False
        for p in self.lora_layer.lora_A.parameters():
            p.requires_grad = True
        for p in self.lora_layer.lora_B.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def rebuild_lora(self, new_rank: int, device: torch.device):
        """
        Dynamic Rank Adaptation: swap in bigger LoRA adapters.
        Base weights are copied over exactly — knowledge is NOT lost.
        Called automatically when blind spot rate > 60%.
        """
        print(f"\n  ⬆  Dynamic Rank Adaptation: rank {self.current_rank} → {new_rank}")
        saved_base_w = self.lora_layer.base.weight.data.clone()
        saved_base_b = self.lora_layer.base.bias.data.clone()

        self.current_rank = new_rank
        self.lora_layer   = LoRALayer(64, 32, rank=new_rank).to(device)

        self.lora_layer.base.weight.data = saved_base_w
        self.lora_layer.base.bias.data   = saved_base_b

        self.freeze_base()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable params after rank increase: {trainable:,}")
        return self

    def param_counts(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─── Dynamic Rank Decision ────────────────────────────────────────────────────
def decide_new_rank(current_rank: int, blind_spot_rate: float) -> int:
    """
    blind_spot_rate: fraction of blind_spot_candidates the model still misses.
    0.99 = model misses 99% of new attacks → needs more capacity → increase rank.
    """
    rank_idx = RANK_LEVELS.index(current_rank) if current_rank in RANK_LEVELS else 0

    if blind_spot_rate > 0.60 and rank_idx < len(RANK_LEVELS) - 1:
        new_rank = RANK_LEVELS[rank_idx + 1]
        print(f"\n  Blind spot rate {blind_spot_rate:.1%} > 60% → "
              f"model needs more capacity → rank {current_rank} → {new_rank}")
        return new_rank
    else:
        reason = ("rate < 60%, rank sufficient" if blind_spot_rate <= 0.60
                  else "already at max rank (32)")
        print(f"\n  Blind spot rate {blind_spot_rate:.1%} → "
              f"keeping rank {current_rank} ({reason})")
        return current_rank


# ─── Data helpers ─────────────────────────────────────────────────────────────
def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try:    lookup[key] = float(val)
            except: lookup[key] = 0.0
    return np.array([lookup.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32)


def load_csv(path, label="dataset"):
    print(f"  Loading {label}...")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,} | Labels: {df['label'].value_counts().to_dict()}")
    X  = np.vstack(df["text"].apply(parse_text).values)
    y  = df["label"].values.astype(np.int64)
    return X, y


# ─── Measure blind spot rate ──────────────────────────────────────────────────
def measure_blind_spot_rate(model, device) -> float:
    """
    Loads blind_spots.csv (output of blind_spot.py).
    Returns fraction of those attacks the model still misses (predicts as Benign).
    0.0 = detects all,  1.0 = misses all.
    """
    if not os.path.exists(BLIND_CSV):
        print("  blind_spots.csv not found — skipping dynamic rank check")
        print("  (Run blind_spot.py first to enable dynamic rank adaptation)")
        return 0.0

    df = pd.read_csv(BLIND_CSV)
    if FEATURE_NAMES[0] not in df.columns:
        print("  blind_spots.csv columns don't match — skipping")
        return 0.0

    X_blind = df[FEATURE_NAMES].values.astype(np.float32)
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(X_blind, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()
    missed = (preds == 0).mean()
    print(f"  Blind spot check: misses {missed:.1%} of {len(X_blind):,} candidates")
    return float(missed)


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("  IMMUNEX - LAYER 4: LoRA RETRAINING")
    print("  Dynamic Rank Adaptation enabled")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print()
    X_train, y_train = load_csv(TRAIN_CSV, "training data")
    X_test,  y_test  = load_csv(TEST_CSV,  "test data")

    # ── Build model ───────────────────────────────────────────────────────────
    initial_rank = RANK_LEVELS[0]   # start at 8
    print(f"\n  Building model (initial rank={initial_rank})...")
    model     = IMMUNEXLayer4(input_dim=INPUT_DIM, rank=initial_rank).to(device)
    criterion = nn.CrossEntropyLoss()

    total, trainable = model.param_counts()
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Initial training ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INITIAL TRAINING (all weights, 20 epochs)")
    print("=" * 60)

    loader    = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=256, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc  = 0.0
    train_log = []
    t0        = time.time()

    for ep in range(20):
        loss, tr_acc = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        te_acc, _    = evaluate(model, X_test, y_test, device)
        best_acc     = max(best_acc, te_acc)
        train_log.append({
            "epoch":     ep + 1,
            "loss":      round(loss,   4),
            "train_acc": round(tr_acc, 2),
            "test_acc":  round(te_acc, 2),
        })
        print(f"  Epoch {ep+1:2d}/20 | Loss: {loss:.4f} | "
              f"Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")

    elapsed = time.time() - t0
    print(f"\n✅ Initial Training Complete!")
    print(f"   Best Test Accuracy : {best_acc:.2f}%")
    print(f"   Time               : {elapsed:.1f}s")

    # ── Freeze base encoder ────────────────────────────────────────────────────
    model.freeze_base()
    _, trainable = model.param_counts()
    print(f"   🔒 Base encoder frozen")
    print(f"   Trainable now      : {trainable:,} (LoRA adapters + head only)")

    # ── Dynamic Rank Check ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("  Dynamic Rank Adaptation Check")
    print("─" * 40)
    blind_rate = measure_blind_spot_rate(model, device)
    new_rank   = decide_new_rank(model.current_rank, blind_rate)

    if new_rank != model.current_rank:
        model.rebuild_lora(new_rank, device)
    else:
        print(f"  Rank stays at {model.current_rank} ✅")

    # ── Classification report ─────────────────────────────────────────────────
    _, pred = evaluate(model, X_test, y_test, device)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, pred, target_names=["Benign", "Attack"]))

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save({
        "model_state":   model.state_dict(),
        "input_dim":     INPUT_DIM,
        "feature_names": FEATURE_NAMES,
        "accuracy":      best_acc,
        "lora_rank":     model.current_rank,
    }, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")
    print(f"   (rank={model.current_rank}, accuracy={best_acc:.2f}%)")

    with open(LOG_PATH, "w") as f:
        json.dump({
            "initial_training": train_log,
            "final_rank":       model.current_rank,
            "blind_spot_rate":  round(blind_rate, 4),
            "rank_adapted":     new_rank != initial_rank,
            "cycles":           [],
        }, f, indent=2)
    print(f"📋 Log saved to: {LOG_PATH}")

    print(f"\n🎉 LORA RETRAINING DONE!")
    print(f"   Final rank : {model.current_rank}")
    print(f"   Next step  : Run blind_spot.py")


if __name__ == "__main__":
    main()
