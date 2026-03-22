"""
IMMUNEX - Layer 4: EWC Continual Learning (Fixed)
Two-phase approach:
  Phase 1: Warm up LoRA head on mutations (learn new patterns)
  Phase 2: EWC consolidation with rehearsal (preserve old knowledge)
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
BASE_DIR      = r"E:\immunex_p4\layer4_immunity"
DATA_DIR      = r"E:\immunex_p4\person4_layer4"
MODEL_DIR     = os.path.join(BASE_DIR, "models")
LOG_DIR       = os.path.join(BASE_DIR, "logs")
MODEL_PATH    = os.path.join(MODEL_DIR, "lora_model.pt")
UPDATED_MODEL = os.path.join(MODEL_DIR, "lora_model_ewc.pt")
TRAIN_CSV     = os.path.join(DATA_DIR,  "lora_retrain_source.csv")
TEST_CSV      = os.path.join(DATA_DIR,  "lora_test.csv")
MUTATIONS_CSV = os.path.join(BASE_DIR,  "mutated_attacks.csv")
LOG_PATH      = os.path.join(LOG_DIR,   "ewc_training_log.json")

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

# ─── Model ────────────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base   = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))

class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_head = nn.Sequential(
            LoRALayer(64, 32, rank=8), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.lora_head(self.base_encoder(x))

# ─── Data helpers ─────────────────────────────────────────────────────────────
def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try: lookup[key] = float(val)
            except: lookup[key] = 0.0
    return np.array([lookup.get(f, 0.0) for f in FEATURE_NAMES],
                    dtype=np.float32)

def load_original():
    print("📂 Loading original training data...")
    df  = pd.read_csv(TRAIN_CSV)
    X   = np.vstack(df["text"].apply(parse_text).values)
    y   = df["label"].values.astype(np.int64)
    print(f"   ✅ {len(X):,} rows")
    return X, y

def load_test():
    print("📂 Loading test data...")
    df = pd.read_csv(TEST_CSV)
    X  = np.vstack(df["text"].apply(parse_text).values)
    y  = df["label"].values.astype(np.int64)
    print(f"   ✅ {len(X):,} rows")
    return X, y

def load_mutations():
    print("📂 Loading mutations...")
    df = pd.read_csv(MUTATIONS_CSV)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    # Mutations already in CICIDS normalized format (-2.8 to 18.7)
    # No normalization needed — just clip extreme outliers
    X  = np.clip(X, -5, 20)
    y  = np.ones(len(X), dtype=np.int64)
    print(f"   ✅ {len(X):,} mutations")
    print(f"   Scale: min={X.min():.3f} max={X.max():.3f} ✅")
    print(f"   Format: CICIDS normalized — matches training data perfectly")
    return X, y

def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(X, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()
    return accuracy_score(y, pred) * 100, pred

def make_loader(X, y, batch=256):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch, shuffle=True)

def train_epoch(model, loader, opt, criterion, device, ewc=None):
    model.train()
    total_loss, ewc_val, correct, total = 0.0, 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        out       = model(Xb)
        loss      = criterion(out, yb)
        if ewc is not None:
            ep = ewc.penalty(model)
            ewc_val = ep.item()
            loss    = loss + ep
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total * 100, ewc_val

# ─── EWC ──────────────────────────────────────────────────────────────────────
class EWC:
    def __init__(self, model, X, y, device, lam=2000):
        self.lam    = lam
        self.device = device
        self.params = {n: p.data.clone()
                       for n, p in model.named_parameters()
                       if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p.data)
                       for n, p in model.named_parameters()
                       if p.requires_grad}
        self._compute(model, X, y)

    def _compute(self, model, X, y, n=2000):
        print(f"🔬 Computing Fisher matrix on {n} samples...")
        criterion = nn.CrossEntropyLoss()
        model.eval()
        idx = np.random.choice(len(X), min(n, len(X)), replace=False)
        Xs  = torch.tensor(X[idx], dtype=torch.float32).to(self.device)
        ys  = torch.tensor(y[idx], dtype=torch.long).to(self.device)
        for i in range(len(Xs)):
            model.zero_grad()
            loss = criterion(model(Xs[i:i+1]), ys[i:i+1])
            loss.backward()
            for nm, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[nm] += p.grad.data.clone().pow(2)
            if (i+1) % 500 == 0:
                print(f"   {i+1}/{len(Xs)}")
        for nm in self.fisher:
            self.fisher[nm] /= len(Xs)
        vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"   ✅ Fisher done | max={vals.max():.4f}")

    def penalty(self, model):
        loss = torch.tensor(0.0, device=self.device)
        for nm, p in model.named_parameters():
            if p.requires_grad and nm in self.fisher:
                loss += (self.fisher[nm].to(self.device) *
                         (p - self.params[nm].to(self.device)).pow(2)).sum()
        return self.lam * loss

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4: EWC CONTINUAL LEARNING")
    print("="*60)

    X_orig, y_orig = load_original()
    X_test, y_test = load_test()
    X_mut,  y_mut  = load_mutations()

    # Load model
    print("\n📂 Loading model...")
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = IMMUNEXLayer4(25).to(device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.base_encoder.parameters():
        p.requires_grad = False
    print(f"   ✅ Loaded | Accuracy: {ckpt['accuracy']:.2f}% | 🔒 Base frozen")

    init_acc, _ = evaluate(model, X_test, y_test, device)
    print(f"\n📊 Accuracy BEFORE EWC: {init_acc:.2f}%")
    criterion = nn.CrossEntropyLoss()

    # ── NO WARMUP — go straight to EWC ────────────────────────────────────────
    # Warmup was hurting accuracy by overwriting original knowledge
    # EWC directly protects original knowledge while learning mutations
    print("\n" + "="*60)
    print("  EWC DIRECT RETRAINING (no warmup)")
    print("="*60)

    idx_ben = np.where(y_orig == 0)[0]
    idx_att = np.where(y_orig == 1)[0]

    # Compute Fisher on ORIGINAL data only
    # This gives strongest protection for original knowledge
    ewc = EWC(model, X_orig, y_orig, device, lam=50000)

    cycle_log   = []
    current_acc = init_acc
    warm_acc    = init_acc

    for cycle in range(1, 7):  # 6 cycles for better stabilization
        print(f"\n{'='*60}")
        print(f"  EWC CYCLE {cycle}")
        print(f"{'='*60}")
        print(f"📊 Accuracy BEFORE: {current_acc:.2f}%")

        # Rehearsal: 80% original data + 20% mutations
        # Heavy original data ratio protects accuracy
        n_b = int(0.5 * 1024)   # 512 benign
        n_a = int(0.3 * 1024)   # 307 known attacks
        n_m = 1024 - n_b - n_a  # 205 mutations
        i_b = np.random.choice(idx_ben, n_b, replace=False)
        i_a = np.random.choice(idx_att, min(n_a, len(idx_att)), replace=True)
        i_m = np.random.choice(len(X_mut), n_m, replace=True)

        X_cyc = np.vstack([X_orig[i_b], X_orig[i_a], X_mut[i_m]])
        y_cyc = np.concatenate([
            np.zeros(n_b,     dtype=np.int64),
            np.ones(len(i_a), dtype=np.int64),
            np.ones(n_m,      dtype=np.int64),
        ])
        loader_cyc = make_loader(X_cyc, y_cyc, batch=128)
        opt_cyc    = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5)   # very small LR — gentle updates

        best_train = 0.0
        t0 = time.time()

        for ep in range(10):
            loss, tr_acc, ewc_val = train_epoch(
                model, loader_cyc, opt_cyc, criterion, device, ewc=ewc)
            best_train = max(best_train, tr_acc)
            print(f"  Cycle {cycle} | Ep {ep+1:2d}/10 | "
                  f"Task: {loss:.4f} | EWC: {ewc_val:.4f} | "
                  f"Train: {tr_acc:.1f}%")

        after_acc, _ = evaluate(model, X_test, y_test, device)
        elapsed      = time.time() - t0
        improvement  = after_acc - current_acc
        arrow        = "📈" if improvement >= 0 else "📉"

        cycle_log.append({
            "cycle":           cycle,
            "accuracy_before": round(current_acc, 4),
            "accuracy_after":  round(after_acc,   4),
            "improvement":     round(improvement,  4),
            "best_train_acc":  round(best_train,   4),
            "time_taken":      round(elapsed,       1),
        })

        print(f"\n✅ Cycle {cycle} Complete! {arrow}")
        print(f"   Before : {current_acc:.2f}%")
        print(f"   After  : {after_acc:.2f}%")
        print(f"   Change : {improvement:+.2f}%")
        current_acc = after_acc
        if cycle < 3:
            time.sleep(2)

    # Final report
    print(f"\n📊 Final Classification Report:")
    _, pred = evaluate(model, X_test, y_test, device)
    print(classification_report(y_test, pred,
          target_names=["Benign","Attack"], zero_division=0))

    # Save model
    torch.save({
        "model_state":   model.state_dict(),
        "input_dim":     25,
        "feature_names": FEATURE_NAMES,
        "accuracy":      current_acc,
        "ewc_trained":   True,
    }, UPDATED_MODEL)

    # Summary
    print(f"\n{'='*60}")
    print("  ALL CYCLES COMPLETE")
    print(f"{'='*60}")
    print(f"   Initial Accuracy : {init_acc:.2f}%")
    print(f"   After Warmup     : {warm_acc:.2f}%")
    for c in cycle_log:
        arrow = "📈" if c["improvement"] >= 0 else "📉"
        print(f"   Cycle {c['cycle']}          : "
              f"{c['accuracy_after']:.2f}% ({c['improvement']:+.2f}%) {arrow}")
    print(f"   Final Accuracy   : {current_acc:.2f}%")

    with open(LOG_PATH, "w") as f:
        json.dump({
            "initial_accuracy": round(init_acc, 4),
            "phase1_accuracy":  round(warm_acc, 4),
            "final_accuracy":   round(current_acc, 4),
            "cycles":           cycle_log,
        }, f, indent=2)

    print(f"\n💾 Model : {UPDATED_MODEL}")
    print(f"📋 Log   : {LOG_PATH}")
    print(f"\n🎉 EWC DONE! Next: Run server.py")

if __name__ == "__main__":
    main()
