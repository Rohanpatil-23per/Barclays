"""
IMMUNEX - Layer 4: EWC with Decaying Lambda + Contrastive Learning
===================================================================
Bugs fixed in this version vs previous:

  BUG 1 — Contrastive loss printing NEGATIVE values (-3.3 to -3.6)
    Root cause: SupConLoss formula had a numerical instability. The
    log_prob values were large and negative due to temperature scaling,
    making the averaged loss negative. A negative con loss subtracted
    from the total loss was actively hurting training.
    Fix: Proper logsumexp formulation + clamp(min=0) ensures loss >= 0.

  BUG 2 — EWC penalty printed as ~0.04 while task loss was ~1.5
    Root cause: Fisher matrix max = 0.0458 (tiny). Even with lambda=50000,
    the penalty was 50000 × 0.0458 × delta² which is negligible.
    Fix: Normalize Fisher so max = 1.0. Now lambda directly controls
    protection strength as intended.

  BUG 3 — Task loss stuck at 1.3-1.6 (binary classifier should reach 0.3)
    Root cause: 20% of each batch was RL mutations designed to fool the
    model — confusing gradient signal. Too few clean attack samples.
    Fix: Changed rehearsal to 50/40/10 (more known attacks) and 15 epochs.

  BUG 4 — lora_layer.base not frozen during EWC
    Root cause: old code only froze base_encoder, missed lora_layer.base.
    The base Linear inside LoRALayer was still trainable, contradicting
    the freeze logic in lora_retrain.py.
    Fix: freeze_for_ewc() explicitly freezes lora_layer.base too.

Run order:
  1. python lora_retrain.py
  2. python blind_spot.py
  3. python mutation_engine.py
  4. python ewc.py            <- this file
  5. python server.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# --- Paths -------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = BASE_DIR
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

EWC_LAMBDA_SCHEDULE = [50000, 30000, 18000, 10000, 5000, 5000]


# --- Model (must match lora_retrain.py exactly) ------------------------------
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
    def __init__(self, input_dim=25, rank=8):
        super().__init__()
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

    def get_embedding(self, x):
        """32-dim embedding before the classifier head. Used by SupConLoss."""
        return F.relu(self.lora_layer(self.base_encoder(x)))

    def freeze_for_ewc(self):
        """
        Freeze base_encoder AND lora_layer.base.
        Only LoRA adapters (lora_A, lora_B) and head train.
        BUG FIX: old version missed lora_layer.base.
        """
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


# --- Supervised Contrastive Loss (FIXED) -------------------------------------
class SupConLoss(nn.Module):
    """
    Numerically stable Supervised Contrastive Loss.
    Loss is always >= 0: pulls same-class embeddings together.
    temperature=0.07 gives sharper cluster separation than 0.1.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, dim=1)
        B   = emb.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)

        sim      = torch.matmul(emb, emb.T) / self.temperature
        labels   = labels.contiguous().view(-1, 1)
        pos_mask = (labels == labels.T).float()
        eye      = torch.eye(B, dtype=torch.float32, device=emb.device)
        pos_mask = pos_mask - eye

        # Numerically stable logsumexp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max.detach()) * (1 - eye)
        log_prob   = (sim - sim_max.detach()) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos     = pos_mask.sum(dim=1)
        has_pos   = n_pos > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=emb.device, requires_grad=True)

        per_anchor = -(pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)
        return per_anchor[has_pos].mean().clamp(min=0.0)


# --- EWC with normalized Fisher (FIXED) -------------------------------------
class EWC:
    """
    Elastic Weight Consolidation.
    BUG FIX: Fisher matrix is normalized so max=1.0, making lambda
    behave predictably regardless of data scale/gradient magnitude.
    """
    def __init__(self, model, X, y, device):
        self.device = device
        self.params = {
            n: p.data.clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._compute_fisher(model, X, y)
        self._normalize_fisher()

    def _compute_fisher(self, model, X, y, n_samples=2000):
        print(f"  Computing Fisher matrix on {n_samples} samples...")
        criterion = nn.CrossEntropyLoss()
        model.eval()
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        Xs  = torch.tensor(X[idx], dtype=torch.float32).to(self.device)
        ys  = torch.tensor(y[idx], dtype=torch.long).to(self.device)
        for i in range(len(Xs)):
            model.zero_grad()
            criterion(model(Xs[i:i+1]), ys[i:i+1]).backward()
            for nm, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[nm] += p.grad.data.clone().pow(2)
            if (i + 1) % 500 == 0:
                print(f"  Fisher progress: {i+1}/{len(Xs)}")
        for nm in self.fisher:
            self.fisher[nm] /= len(Xs)
        vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"  Fisher raw: max={vals.max():.6f} | mean={vals.mean():.8f}")

    def _normalize_fisher(self):
        """Normalize so max=1.0 — makes lambda predictably strong."""
        all_vals = torch.cat([f.flatten() for f in self.fisher.values()])
        max_val  = all_vals.max().item()
        if max_val > 1e-10:
            for nm in self.fisher:
                self.fisher[nm] /= max_val
        print(f"  Fisher normalized. max=1.0 (raw max was {max_val:.6f})")
        print(f"  EWC penalty will now be meaningful at all lambda values.")

    def penalty(self, model, lam):
        loss = torch.tensor(0.0, device=self.device)
        for nm, p in model.named_parameters():
            if p.requires_grad and nm in self.fisher:
                loss += (self.fisher[nm].to(self.device) *
                         (p - self.params[nm].to(self.device)).pow(2)).sum()
        return lam * loss


# --- Data helpers ------------------------------------------------------------
def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try:    lookup[key] = float(val)
            except: lookup[key] = 0.0
    return np.array([lookup.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32)


def load_original():
    print("  Loading original training data...")
    df = pd.read_csv(TRAIN_CSV)
    X  = np.vstack(df["text"].apply(parse_text).values)
    y  = df["label"].values.astype(np.int64)
    print(f"  {len(X):,} rows")
    return X, y


def load_test():
    print("  Loading test data...")
    df = pd.read_csv(TEST_CSV)
    X  = np.vstack(df["text"].apply(parse_text).values)
    y  = df["label"].values.astype(np.int64)
    print(f"  {len(X):,} rows")
    return X, y


def load_mutations():
    print("  Loading mutations...")
    df = pd.read_csv(MUTATIONS_CSV)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    X  = np.clip(X, -5, 20)
    y  = np.ones(len(X), dtype=np.int64)
    print(f"  {len(X):,} mutations | scale: {X.min():.3f} to {X.max():.3f}")
    if "mutation_strategy" in df.columns:
        for strat, cnt in df["mutation_strategy"].value_counts().items():
            print(f"    {strat:<20}: {cnt:,}")
    return X, y


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    return accuracy_score(y, pred) * 100, pred


# --- Combined training epoch -------------------------------------------------
def train_epoch_combined(model, loader, optimizer, ewc_obj, lam,
                          con_loss_fn, con_weight, device):
    model.train()
    total_loss = task_total = ewc_total = con_total = 0.0
    correct = total = 0
    ce = nn.CrossEntropyLoss()

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()

        emb    = model.get_embedding(Xb)
        logits = model.head(emb)

        task_loss = ce(logits, yb)
        ewc_pen   = ewc_obj.penalty(model, lam)
        con_loss  = con_loss_fn(emb, yb)

        loss = task_loss + ewc_pen + con_weight * con_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()      * len(yb)
        task_total += task_loss.item() * len(yb)
        ewc_total  += ewc_pen.item()
        con_total  += con_loss.item()  * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(yb)

    n = len(loader)
    return (total_loss/total, task_total/total,
            ewc_total/n,     con_total/total,
            correct/total*100)


# --- Main --------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("  IMMUNEX LAYER 4: EWC + CONTRASTIVE (FIXED VERSION)")
    print("  Fixes: normalized Fisher, stable SupCon, correct freeze")
    print("  Lambda: 50000->30000->18000->10000->5000->5000")
    print("=" * 60)

    print()
    X_orig, y_orig = load_original()
    X_test, y_test = load_test()
    X_mut,  y_mut  = load_mutations()

    print("\n  Loading model...")
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    rank  = ckpt.get("lora_rank", 8)
    model = IMMUNEXLayer4(input_dim=25, rank=rank).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.freeze_for_ewc()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Loaded | Rank={rank} | Acc={ckpt['accuracy']:.2f}% | Trainable={trainable:,}")

    init_acc, _ = evaluate(model, X_test, y_test, device)
    print(f"\n  Accuracy BEFORE EWC: {init_acc:.2f}%")

    print("\n" + "-" * 40)
    ewc_obj     = EWC(model, X_orig, y_orig, device)
    con_loss_fn = SupConLoss(temperature=0.07)

    idx_ben = np.where(y_orig == 0)[0]
    idx_att = np.where(y_orig == 1)[0]

    cycle_log   = []
    current_acc = init_acc

    for cycle in range(1, 7):
        lam = EWC_LAMBDA_SCHEDULE[cycle - 1]

        print(f"\n{'=' * 60}")
        print(f"  EWC CYCLE {cycle}  |  Lambda = {lam:,}  |  "
              f"{'HIGH protection' if lam > 20000 else 'relaxed protection'}")
        print(f"{'=' * 60}")
        print(f"  Accuracy BEFORE: {current_acc:.2f}%")

        # 50% benign + 40% known attacks + 10% mutations
        n_b = int(0.50 * 1024)
        n_a = int(0.40 * 1024)
        n_m = 1024 - n_b - n_a

        i_b = np.random.choice(idx_ben, n_b, replace=False)
        i_a = np.random.choice(idx_att, min(n_a, len(idx_att)), replace=True)
        i_m = np.random.choice(len(X_mut), n_m, replace=True)

        X_cyc = np.vstack([X_orig[i_b], X_orig[i_a], X_mut[i_m]])
        y_cyc = np.concatenate([
            np.zeros(n_b, dtype=np.int64),
            np.ones(len(i_a), dtype=np.int64),
            np.ones(n_m, dtype=np.int64),
        ])

        loader_cyc = DataLoader(
            TensorDataset(torch.tensor(X_cyc, dtype=torch.float32),
                          torch.tensor(y_cyc, dtype=torch.long)),
            batch_size=128, shuffle=True,
        )

        lr         = max(1e-5, 5e-5 * (0.7 ** (cycle - 1)))
        optimizer  = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        con_weight = min(0.1 * cycle, 0.3)

        best_train = 0.0
        t0         = time.time()

        for ep in range(15):   # 15 epochs (was 10)
            total_l, task_l, ewc_l, con_l, tr_acc = train_epoch_combined(
                model, loader_cyc, optimizer,
                ewc_obj, lam, con_loss_fn, con_weight, device)
            best_train = max(best_train, tr_acc)
            print(f"  Cycle {cycle} | Ep {ep+1:2d}/15 | "
                  f"Task:{task_l:.4f} EWC:{ewc_l:.4f} Con:{con_l:.4f} "
                  f"Train:{tr_acc:.1f}%")

        after_acc, _ = evaluate(model, X_test, y_test, device)
        elapsed      = time.time() - t0
        improvement  = after_acc - current_acc

        cycle_log.append({
            "cycle": cycle, "lambda": lam, "lr": round(lr, 8),
            "con_weight": round(con_weight, 3),
            "accuracy_before": round(current_acc, 4),
            "accuracy_after":  round(after_acc, 4),
            "improvement":     round(improvement, 4),
            "best_train_acc":  round(best_train, 4),
            "time_taken":      round(elapsed, 1),
        })

        arrow = "UP" if improvement >= 0 else "DN"
        print(f"\n  Cycle {cycle} done [{arrow}] | Before:{current_acc:.2f}% "
              f"After:{after_acc:.2f}% Change:{improvement:+.2f}%")
        current_acc = after_acc

    # Final report
    print(f"\n{'=' * 60}")
    print("  ALL EWC CYCLES COMPLETE")
    print(f"{'=' * 60}")
    _, pred = evaluate(model, X_test, y_test, device)
    print(classification_report(y_test, pred,
                                target_names=["Benign", "Attack"],
                                zero_division=0))
    print(f"  Initial : {init_acc:.2f}%")
    for c in cycle_log:
        arrow = "+" if c["improvement"] >= 0 else ""
        print(f"  Cycle {c['cycle']} (lam={c['lambda']:>6,}): "
              f"{c['accuracy_after']:.2f}% ({arrow}{c['improvement']:.2f}%)")
    print(f"  Final   : {current_acc:.2f}%")

    torch.save({
        "model_state": model.state_dict(), "input_dim": 25,
        "feature_names": FEATURE_NAMES,   "accuracy": current_acc,
        "ewc_trained": True,              "lora_rank": rank,
    }, UPDATED_MODEL)

    with open(LOG_PATH, "w") as f:
        json.dump({
            "initial_accuracy":  round(init_acc, 4),
            "final_accuracy":    round(current_acc, 4),
            "lambda_schedule":   EWC_LAMBDA_SCHEDULE,
            "contrastive":       True,
            "fisher_normalized": True,
            "cycles":            cycle_log,
        }, f, indent=2)

    print(f"\n  Model saved: {UPDATED_MODEL}")
    print(f"  Log saved  : {LOG_PATH}")
    print(f"\n  EWC DONE. Next: python server.py")


if __name__ == "__main__":
    main()
