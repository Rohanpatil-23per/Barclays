"""
IMMUNEX - Layer 4: EWC with Decaying Lambda + Contrastive Learning
===================================================================
What this file does:
  Retrains the model on 30,000 mutations (from mutation_engine.py)
  without forgetting what it learned during initial training.
  Uses two mechanisms together:

  1. Elastic Weight Consolidation (EWC) — prevents forgetting
     - After loading the model, computes a Fisher Information Matrix
       which gives every weight an "importance score"
     - High importance = changing this weight hurts accuracy → protect it
     - Low importance  = safe to change → allow it to adapt
     - Protection formula:
         Total Loss = Task Loss + lambda × Fisher × (weight_change)²
     - Lambda DECAYS each cycle:
         Cycle 1: 50,000  → strong protection (model adjusting)
         Cycle 2: 30,000  → loosen slightly
         Cycle 3: 18,000  → more room to learn new patterns
         Cycle 4: 10,000  → balanced
         Cycle 5+: 5,000  → floor (Fisher matrix still guides protection)
       Why decay: Early cycles need strong protection. Later cycles can
       loosen because the Fisher matrix itself already knows which
       specific weights are critical — the large lambda is no longer needed.

  2. Supervised Contrastive Learning (SupCon) — sharpens decision boundary
     - Pulls Benign embeddings together, pushes Attack embeddings apart
     - Cleaner class clusters = model learns "what fundamentally makes
       an attack different from benign" rather than memorising patterns
     - This makes EWC's job easier: it only needs to protect the boundary,
       not every individual memorised pattern
     - Con weight ramps 0.1 → 0.3 across cycles as model stabilises

  Rehearsal batch per cycle (1024 samples total):
     50% Benign originals  → never forget what normal looks like
     30% Known attacks     → never forget original attack signatures
     20% New mutations     → learn new attack variants

Run order:
  1. python lora_retrain.py
  2. python blind_spot.py
  3. python mutation_engine.py
  4. python ewc.py            ← this file
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

# ─── Paths ────────────────────────────────────────────────────────────────────
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

# Lambda decays each cycle — starts high, relaxes as Fisher guides protection
EWC_LAMBDA_SCHEDULE = [50000, 30000, 18000, 10000, 5000, 5000]


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
        """
        Returns 32-dim embedding before the final classifier head.
        Used by SupConLoss to compute class-separation in embedding space.
        """
        return F.relu(self.lora_layer(self.base_encoder(x)))


# ─── Supervised Contrastive Loss ─────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    For a batch of (embedding, label) pairs:
      - Pulls same-class embeddings closer together (positive pairs)
      - Pushes different-class embeddings further apart (negative pairs)

    After training with SupCon:
      Benign embeddings cluster in one region of 32-dim space
      Attack embeddings cluster in another region
      Decision boundary becomes sharper and more robust

    temperature=0.1: lower = sharper separation, 0.1 is standard practice.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings, dim=1)   # L2-normalize → cosine similarity
        B   = emb.shape[0]

        sim        = torch.matmul(emb, emb.T) / self.temperature   # (B, B)
        labels     = labels.view(-1, 1)
        pos_mask   = (labels == labels.T).float()
        self_mask  = torch.eye(B, device=emb.device)
        pos_mask   = pos_mask - self_mask
        neg_mask   = 1 - pos_mask - self_mask  # noqa: F841 (kept for clarity)

        exp_sim    = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        exp_sim    = exp_sim * (1 - self_mask)

        log_prob   = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos      = pos_mask.sum(dim=1)
        has_pos    = n_pos > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=emb.device)

        loss = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        return loss[has_pos].mean()


# ─── EWC ──────────────────────────────────────────────────────────────────────
class EWC:
    """
    Elastic Weight Consolidation.

    Computes Fisher Information Matrix once on original training data.
    During each retraining cycle, adds penalty:
      penalty = lambda × Σ  Fisher[w] × (w_current - w_original)²

    lambda comes from EWC_LAMBDA_SCHEDULE and decays each cycle.
    High Fisher + big weight change → huge penalty → optimizer avoids it.
    """
    def __init__(self, model, X, y, device):
        self.device = device
        self.params = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._compute_fisher(model, X, y)

    def _compute_fisher(self, model, X, y, n_samples=2000):
        print(f"  Computing Fisher matrix on {n_samples} samples...")
        criterion = nn.CrossEntropyLoss()
        model.eval()
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        Xs  = torch.tensor(X[idx], dtype=torch.float32).to(self.device)
        ys  = torch.tensor(y[idx], dtype=torch.long).to(self.device)

        for i in range(len(Xs)):
            model.zero_grad()
            loss = criterion(model(Xs[i:i+1]), ys[i:i+1])
            loss.backward()
            for nm, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[nm] += p.grad.data.clone().pow(2)
            if (i + 1) % 500 == 0:
                print(f"  Fisher progress: {i+1}/{len(Xs)}")

        for nm in self.fisher:
            self.fisher[nm] /= len(Xs)

        vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"  ✅ Fisher done | max={vals.max():.4f} | mean={vals.mean():.6f}")

    def penalty(self, model, lam: float) -> torch.Tensor:
        """EWC penalty with variable lambda (decays each cycle)."""
        loss = torch.tensor(0.0, device=self.device)
        for nm, p in model.named_parameters():
            if p.requires_grad and nm in self.fisher:
                loss += (
                    self.fisher[nm].to(self.device) *
                    (p - self.params[nm].to(self.device)).pow(2)
                ).sum()
        return lam * loss


# ─── Data helpers ─────────────────────────────────────────────────────────────
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
    print(f"  ✅ {len(X):,} rows")
    return X, y


def load_test():
    print("  Loading test data...")
    df = pd.read_csv(TEST_CSV)
    X  = np.vstack(df["text"].apply(parse_text).values)
    y  = df["label"].values.astype(np.int64)
    print(f"  ✅ {len(X):,} rows")
    return X, y


def load_mutations():
    print("  Loading mutations...")
    df = pd.read_csv(MUTATIONS_CSV)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    X  = np.clip(X, -5, 20)
    y  = np.ones(len(X), dtype=np.int64)
    print(f"  ✅ {len(X):,} mutations | scale: {X.min():.3f} to {X.max():.3f}")

    # Print breakdown by strategy if available
    if "mutation_strategy" in df.columns:
        print(f"  Strategy breakdown:")
        for strat, cnt in df["mutation_strategy"].value_counts().items():
            print(f"    {strat:<20}: {cnt:,}")
    return X, y


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(X, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()
    return accuracy_score(y, pred) * 100, pred


# ─── Combined training epoch ──────────────────────────────────────────────────
def train_epoch_combined(model, loader, optimizer,
                          ewc, lam, con_loss_fn, con_weight, device):
    """
    One epoch of combined loss:
      Total = CrossEntropy(task) + EWC_penalty(lam) + con_weight × SupCon

    Prints Task / EWC / Con loss separately so you can see each component.
    """
    model.train()
    total_loss = task_total = ewc_total = con_total = 0.0
    correct = total = 0
    ce_criterion = nn.CrossEntropyLoss()

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()

        embeddings = model.get_embedding(Xb)   # (B, 32) for SupCon
        logits     = model.head(embeddings)    # (B, 2)  for CrossEntropy

        task_loss = ce_criterion(logits, yb)
        ewc_pen   = ewc.penalty(model, lam)
        con_loss  = con_loss_fn(embeddings, yb)

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
    return (total_loss / total,
            task_total / total,
            ewc_total  / n,
            con_total  / total,
            correct    / total * 100)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("  IMMUNEX - LAYER 4: EWC + CONTRASTIVE LEARNING")
    print("  Lambda schedule: 50000→30000→18000→10000→5000→5000")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print()
    X_orig, y_orig = load_original()
    X_test, y_test = load_test()
    X_mut,  y_mut  = load_mutations()   # noqa: F841

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n  Loading model...")
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    rank  = ckpt.get("lora_rank", 8)   # reads rank set by lora_retrain.py
    model = IMMUNEXLayer4(input_dim=25, rank=rank).to(device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.base_encoder.parameters():
        p.requires_grad = False
    print(f"  ✅ Loaded | Rank={rank} | Accuracy: {ckpt['accuracy']:.2f}% | 🔒 Base frozen")

    init_acc, _ = evaluate(model, X_test, y_test, device)
    print(f"\n  Accuracy BEFORE EWC: {init_acc:.2f}%")

    # ── Compute Fisher matrix ─────────────────────────────────────────────────
    print("\n" + "─" * 40)
    ewc_obj     = EWC(model, X_orig, y_orig, device)
    con_loss_fn = SupConLoss(temperature=0.1)

    # Index splits for rehearsal batches
    idx_ben = np.where(y_orig == 0)[0]
    idx_att = np.where(y_orig == 1)[0]

    cycle_log   = []
    current_acc = init_acc

    # ── 6 EWC cycles with decaying lambda ─────────────────────────────────────
    for cycle in range(1, 7):
        lam = EWC_LAMBDA_SCHEDULE[cycle - 1]

        print(f"\n{'=' * 60}")
        print(f"  EWC CYCLE {cycle}  |  Lambda = {lam:,}")
        protection = "protection HIGH" if lam > 20000 else "protection relaxed"
        print(f"  ({protection} — lambda decays as Fisher guides protection)")
        print(f"{'=' * 60}")
        print(f"  Accuracy BEFORE: {current_acc:.2f}%")

        # Rehearsal batch: 50% benign + 30% attacks + 20% mutations
        n_b = int(0.50 * 1024)
        n_a = int(0.30 * 1024)
        n_m = 1024 - n_b - n_a

        i_b  = np.random.choice(idx_ben, n_b, replace=False)
        i_a  = np.random.choice(idx_att, min(n_a, len(idx_att)), replace=True)
        i_m  = np.random.choice(len(X_mut), n_m, replace=True)

        X_cyc = np.vstack([X_orig[i_b], X_orig[i_a], X_mut[i_m]])
        y_cyc = np.concatenate([
            np.zeros(n_b,     dtype=np.int64),
            np.ones(len(i_a), dtype=np.int64),
            np.ones(n_m,      dtype=np.int64),
        ])

        loader_cyc = DataLoader(
            TensorDataset(
                torch.tensor(X_cyc, dtype=torch.float32),
                torch.tensor(y_cyc, dtype=torch.long),
            ),
            batch_size=128, shuffle=True,
        )

        # LR also decays with lambda for gentler updates in later cycles
        lr        = max(1e-5, 5e-5 * (0.7 ** (cycle - 1)))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

        # Contrastive weight ramps up: light early, stronger later
        con_weight = min(0.1 * cycle, 0.3)

        best_train = 0.0
        t0         = time.time()

        for ep in range(10):
            total_l, task_l, ewc_l, con_l, tr_acc = train_epoch_combined(
                model, loader_cyc, optimizer,
                ewc_obj, lam, con_loss_fn, con_weight, device,
            )
            best_train = max(best_train, tr_acc)
            print(f"  Cycle {cycle} | Ep {ep+1:2d}/10 | "
                  f"Task: {task_l:.4f} | EWC: {ewc_l:.4f} | "
                  f"Con: {con_l:.4f} | Train: {tr_acc:.1f}%")

        after_acc, _ = evaluate(model, X_test, y_test, device)
        elapsed      = time.time() - t0
        improvement  = after_acc - current_acc
        arrow        = "📈" if improvement >= 0 else "📉"

        cycle_log.append({
            "cycle":           cycle,
            "lambda":          lam,
            "lr":              round(lr, 8),
            "con_weight":      round(con_weight, 3),
            "accuracy_before": round(current_acc, 4),
            "accuracy_after":  round(after_acc,   4),
            "improvement":     round(improvement,  4),
            "best_train_acc":  round(best_train,   4),
            "time_taken":      round(elapsed,       1),
        })

        next_lam = EWC_LAMBDA_SCHEDULE[min(cycle, 5)]
        print(f"\n✅ Cycle {cycle} Complete {arrow}")
        print(f"   Lambda    : {lam:,}  →  next: {next_lam:,}")
        print(f"   Before    : {current_acc:.2f}%")
        print(f"   After     : {after_acc:.2f}%")
        print(f"   Change    : {improvement:+.2f}%")
        current_acc = after_acc

    # ── Final report ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ALL EWC CYCLES COMPLETE")
    print(f"{'=' * 60}")

    _, pred = evaluate(model, X_test, y_test, device)
    print(f"\n📊 Final Classification Report:")
    print(classification_report(y_test, pred,
                                target_names=["Benign", "Attack"],
                                zero_division=0))

    print(f"  Initial Accuracy : {init_acc:.2f}%")
    for c in cycle_log:
        arrow = "📈" if c["improvement"] >= 0 else "📉"
        print(f"  Cycle {c['cycle']} (λ={c['lambda']:>6,}) : "
              f"{c['accuracy_after']:.2f}%  ({c['improvement']:+.2f}%) {arrow}")
    print(f"  Final Accuracy   : {current_acc:.2f}%")

    # ── Save model ─────────────────────────────────────────────────────────────
    torch.save({
        "model_state":   model.state_dict(),
        "input_dim":     25,
        "feature_names": FEATURE_NAMES,
        "accuracy":      current_acc,
        "ewc_trained":   True,
        "lora_rank":     rank,
    }, UPDATED_MODEL)

    with open(LOG_PATH, "w") as f:
        json.dump({
            "initial_accuracy": round(init_acc, 4),
            "final_accuracy":   round(current_acc, 4),
            "lambda_schedule":  EWC_LAMBDA_SCHEDULE,
            "contrastive":      True,
            "cycles":           cycle_log,
        }, f, indent=2)

    print(f"\n💾 Model : {UPDATED_MODEL}")
    print(f"📋 Log   : {LOG_PATH}")
    print(f"\n🎉 EWC DONE! Next: Run server.py")


if __name__ == "__main__":
    main()
