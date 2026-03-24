import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# ─────────────────────────────────────────────────────────────
# WHY BiLSTM:
# GATv2 looks at alerts spatially (who relates to whom).
# BiLSTM looks at alerts TEMPORALLY (what happened first,
# second, third — and what comes next).
#
# Bidirectional = reads sequence both forward AND backward:
# Forward  : what caused the current state
# Backward : what this state leads to
#
# Two output heads:
# 1. stage_classifier → current MITRE stage (what is now)
# 2. next_stage_head  → predicted next stage (what comes next)
#
# KEY FIXES IN THIS VERSION:
# 1. seq_len=5, step=1 — allows rare classes to form sequences
# 2. Both train AND val balanced — honest macro F1 evaluation
# 3. Impact capped at same level as other attack classes
# 4. min_samples=8000 — more training variety per stage
# ─────────────────────────────────────────────────────────────

LABEL_TO_MITRE = {
    'Benign'                     : 'Benign',
    'Normal'                     : 'Benign',
    'PortScan'                   : 'Reconnaissance',
    'Reconnaissance'             : 'Reconnaissance',
    'Fuzzers'                    : 'Reconnaissance',
    'Analysis'                   : 'Reconnaissance',
    'FTP-Patator'                : 'Initial_Access',
    'SSH-Patator'                : 'Initial_Access',
    'Exploits'                   : 'Initial_Access',
    'Shellcode'                  : 'Initial_Access',
    'Bot'                        : 'Execution',
    'Backdoor'                   : 'Execution',
    'Worms'                      : 'Execution',
    'DDoS'                       : 'Impact',
    'DoS Hulk'                   : 'Impact',
    'DoS GoldenEye'              : 'Impact',
    'DoS slowloris'              : 'Impact',
    'DoS Slowhttptest'           : 'Impact',
    'Generic'                    : 'Impact',
    'DoS'                        : 'Impact',
    'Infiltration'               : 'Exfiltration',
    'Heartbleed'                 : 'Exploitation',
    'Web Attack - Brute Force'   : 'Initial_Access',
    'Web Attack - XSS'           : 'Initial_Access',
    'Web Attack - Sql Injection' : 'Initial_Access',
}

MITRE_TO_ID = {
    'Benign'         : 0,
    'Reconnaissance' : 1,
    'Initial_Access' : 2,
    'Execution'      : 3,
    'Impact'         : 4,
    'Exfiltration'   : 5,
    'Exploitation'   : 6,
}

ID_TO_MITRE = {v: k for k, v in MITRE_TO_ID.items()}
NUM_STAGES  = len(MITRE_TO_ID)  # 7

FEATURE_COLS = [
    'flow_duration',
    'syn_flag_count',
    'fin_flag_count',
    'rst_flag_count',
    'flow_bytes_s',
    'flow_packets_s'
]


# ─────────────────────────────────────────────────────────────
# FUNCTION 1 — Fix label encoding
# ─────────────────────────────────────────────────────────────

def fix_labels(label):
    """Fix web attack encoding issues by keyword detection."""
    if not isinstance(label, str):
        return label
    l = label.lower()
    if 'brute force' in l and 'web' in l:
        return 'Web Attack - Brute Force'
    if 'xss' in l:
        return 'Web Attack - XSS'
    if 'sql' in l:
        return 'Web Attack - Sql Injection'
    return label


# ─────────────────────────────────────────────────────────────
# FUNCTION 2 — Dataset class
# ─────────────────────────────────────────────────────────────

class AlertSequenceDataset(Dataset):
    """
    Wraps sequences for PyTorch DataLoader.

    Each item:
    - sequence   : (seq_len=5, features=6) tensor
    - label      : current MITRE stage ID (0-6)
    - next_label : predicted next MITRE stage ID (0-6)
    """
    def __init__(self, sequences, labels, next_labels):
        self.sequences   = sequences
        self.labels      = labels
        self.next_labels = next_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx],
                         dtype=torch.float32),
            torch.tensor(self.labels[idx],
                         dtype=torch.long),
            torch.tensor(self.next_labels[idx],
                         dtype=torch.long)
        )


# ─────────────────────────────────────────────────────────────
# FUNCTION 3 — BiLSTM Model
# ─────────────────────────────────────────────────────────────

class AttackSequenceBiLSTM(nn.Module):
    """
    Bidirectional LSTM for attack stage classification
    and next stage prediction.

    Architecture:
    Input  : (batch, seq_len=5, features=6)
    BiLSTM : 2 layers, 128 hidden units per direction
    Output : 256-dim state (128 fwd + 128 bwd)
    Head 1 : stage_classifier  → 7 MITRE stage probs
    Head 2 : next_stage_head   → 7 next stage probs
    """
    def __init__(self, input_size=6, hidden_size=128,
                 num_layers=2, num_stages=7, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout
        )

        lstm_output_size = hidden_size * 2  # 256

        # Batch norm stabilizes sequential model training
        self.bn = nn.BatchNorm1d(lstm_output_size)

        # Head 1: What MITRE stage is happening NOW?
        self.stage_classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_stages)
        )

        # Head 2: What MITRE stage comes NEXT?
        # Person 3's DQN uses this for proactive defense
        self.next_stage_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_stages)
        )

    def forward(self, x):
        """
        x shape: (batch, seq_len=5, input_size=6)
        """
        out, (h, c) = self.lstm(x)

        # Last timestep captures full sequence context
        last_out = out[:, -1, :]
        last_out = self.bn(last_out)

        stage_logits      = self.stage_classifier(last_out)
        next_stage_logits = self.next_stage_head(last_out)

        return stage_logits, next_stage_logits


# ─────────────────────────────────────────────────────────────
# FUNCTION 4 — Training and evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    """
    One pass through training data.
    Loss = stage_loss + next_stage_loss
    Both heads train simultaneously sharing LSTM backbone.
    """
    model.train()
    total_loss = 0

    for seqs, labels, next_labels in loader:
        seqs        = seqs.to(device)
        labels      = labels.to(device)
        next_labels = next_labels.to(device)

        optimizer.zero_grad()

        stage_out, next_out = model(seqs)

        stage_loss = F.cross_entropy(stage_out, labels)
        next_loss  = F.cross_entropy(next_out, next_labels)
        loss       = stage_loss + next_loss

        loss.backward()

        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates both heads on balanced validation set.

    Returns:
    - stage_acc : accuracy on current stage prediction
    - next_acc  : accuracy on next stage prediction
    - stage_f1  : macro F1 across all 7 MITRE stages
                  PRIMARY METRIC — all stages weighted equally
    """
    model.eval()
    stage_correct = 0
    next_correct  = 0
    total         = 0
    all_preds     = []
    all_labels    = []

    with torch.no_grad():
        for seqs, labels, next_labels in loader:
            seqs        = seqs.to(device)
            labels      = labels.to(device)
            next_labels = next_labels.to(device)

            stage_out, next_out = model(seqs)

            stage_preds = stage_out.argmax(dim=1)
            next_preds  = next_out.argmax(dim=1)

            stage_correct += (stage_preds == labels).sum().item()
            next_correct  += (next_preds == next_labels).sum().item()
            total         += labels.size(0)

            all_preds.extend(stage_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    stage_acc = stage_correct / total
    next_acc  = next_correct  / total
    stage_f1  = f1_score(
        all_labels, all_preds,
        average='macro', zero_division=0
    )

    return stage_acc, next_acc, stage_f1


# ─────────────────────────────────────────────────────────────
# FUNCTION 5 — Build sequences
# seq_len=5, step=1 maximizes rare class sequence extraction
# ─────────────────────────────────────────────────────────────

def prepare_sequences(df, seq_len=5, step=1):
    """
    Sliding window over alert rows.

    seq_len=5 : 5 alerts per window
                Exfiltration (36 rows) → ~31 sequences
                Exploitation (11 rows) → ~6 sequences
                With seq_len=20 both got ZERO sequences

    step=1    : maximum sequence extraction from limited data

    Label     = most common MITRE stage in current window
    Next label = most common stage in next window
    """
    sequences   = []
    labels      = []
    next_labels = []

    feature_vals = df[FEATURE_COLS].values
    mitre_ids    = df['mitre_id'].values
    total        = len(feature_vals)

    for start in range(0, total - seq_len, step):
        end      = start + seq_len
        next_end = end   + seq_len

        seq         = feature_vals[start:end]
        window_lbls = mitre_ids[start:end]
        label       = int(np.bincount(window_lbls).argmax())

        if next_end <= total:
            next_window = mitre_ids[end:next_end]
            next_label  = int(np.bincount(next_window).argmax())
        else:
            next_label = label

        sequences.append(seq)
        labels.append(label)
        next_labels.append(next_label)

    return (np.array(sequences),
            np.array(labels),
            np.array(next_labels))


# ─────────────────────────────────────────────────────────────
# FUNCTION 6 — Balance sequences
# Used for BOTH train and val sets
# ─────────────────────────────────────────────────────────────

def balance_sequences(sequences, labels, next_labels,
                      min_samples=5000):
    """
    Three-step balancing:
    1. Oversample all rare attack stages to min_samples
    2. Cap Impact at min_samples (prevent domination)
    3. Cap Benign at 3x min_samples

    Applied to BOTH train and val:
    - Train: ensures model learns all 7 stages equally
    - Val  : ensures macro F1 reflects per-stage performance
             honestly, not dominated by Benign/Impact counts
    Balancing val does NOT cause leakage — val rows are
    never used for training regardless of how they're sampled.
    """
    unique_labels = np.unique(labels)

    balanced_seq  = []
    balanced_lbl  = []
    balanced_next = []

    for lbl in unique_labels:
        mask       = labels == lbl
        lbl_seqs   = sequences[mask]
        lbl_lbls   = labels[mask]
        lbl_next   = next_labels[mask]
        stage_name = ID_TO_MITRE.get(lbl, str(lbl))
        original_n = mask.sum()

        if stage_name == 'Benign':
            # Cap Benign at 3x min_samples
            cap = min_samples * 3
            n   = min(cap, len(lbl_seqs))
            idx = np.random.choice(
                len(lbl_seqs), size=n, replace=False
            )
            lbl_seqs = lbl_seqs[idx]
            lbl_lbls = lbl_lbls[idx]
            lbl_next = lbl_next[idx]
            print(f"  {stage_name}: {original_n:,} → {n:,} "
                  f"(capped at 3x)")

        elif stage_name == 'Impact':
            # Cap Impact at min_samples to prevent domination
            n = min(min_samples, len(lbl_seqs))
            if len(lbl_seqs) >= n:
                idx = np.random.choice(
                    len(lbl_seqs), size=n, replace=False
                )
            else:
                idx = np.random.choice(
                    len(lbl_seqs), size=n, replace=True
                )
            lbl_seqs = lbl_seqs[idx]
            lbl_lbls = lbl_lbls[idx]
            lbl_next = lbl_next[idx]
            print(f"  {stage_name}: {original_n:,} → {n:,} "
                  f"(capped at min_samples)")

        else:
            # Oversample all other attack stages to min_samples
            if len(lbl_seqs) < min_samples:
                idx = np.random.choice(
                    len(lbl_seqs), size=min_samples, replace=True
                )
                lbl_seqs = lbl_seqs[idx]
                lbl_lbls = lbl_lbls[idx]
                lbl_next = lbl_next[idx]
                print(f"  {stage_name}: {original_n} → "
                      f"{min_samples} (oversampled)")
            else:
                print(f"  {stage_name}: {original_n:,} (kept)")

        balanced_seq.append(lbl_seqs)
        balanced_lbl.append(lbl_lbls)
        balanced_next.append(lbl_next)

    final_seqs  = np.concatenate(balanced_seq)
    final_lbls  = np.concatenate(balanced_lbl)
    final_nexts = np.concatenate(balanced_next)

    print(f"  Total: {len(final_seqs):,} sequences")
    print(f"  Distribution:")
    unique, counts = np.unique(final_lbls, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {ID_TO_MITRE[u]}: {c:,}")

    return final_seqs, final_lbls, final_nexts


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ── Load dataset ──────────────────────────────────────────
    print("Loading BiLSTM dataset...")
    BILSTM_FILE = (
    'data/raw/new_dataset/team_datasets/'
    'person2_layer2/bilstm_cicids_augmented.csv'
    )

    df = pd.read_csv(BILSTM_FILE)
    print(f"  Rows: {len(df):,}")

    df['label']       = df['label'].apply(fix_labels)
    df['mitre_stage'] = df['label'].map(LABEL_TO_MITRE)
    df['mitre_stage'] = df['mitre_stage'].fillna('Benign')
    df['mitre_id']    = df['mitre_stage'].map(MITRE_TO_ID)

    print(f"\nMITRE Distribution:")
    print(df['mitre_stage'].value_counts())

    # ── Scale features ────────────────────────────────────────
    print("\nScaling features...")
    df[FEATURE_COLS] = df[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    with open('models/bilstm/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  Scaler saved")

    # ── Sort by MITRE stage for sequence building ─────────────
    # Grouping same-stage rows together ensures sliding window
    # creates pure attack-type sequences for rare classes
    # Exfiltration rows are consecutive → 31 clean sequences
    # Exploitation rows are consecutive → 6 clean sequences
    print("\nSorting by MITRE stage for sequence building...")
    df = df.sort_values('mitre_id').reset_index(drop=True)
    print("  Sorted — rare class rows now consecutive")

    # ── Build sequences ───────────────────────────────────────
    print("Building sequences (seq_len=5, step=1)...")
    sequences, labels, next_labels = prepare_sequences(
        df, seq_len=5, step=1
    )
    print(f"  Total sequences: {len(sequences):,}")

    print(f"\nSequence stage distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {ID_TO_MITRE[u]}: {c:,}")

    # ── Train/Val split ───────────────────────────────────────
    # Stratified split ensures all stages in both sets
    print("\nSplitting sequences 80/20...")
    (X_train, X_val,
     y_train, y_val,
     yn_train, yn_val) = train_test_split(
        sequences, labels, next_labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ── Balance TRAINING sequences ────────────────────────────
    # min_samples=8000: more variety per stage than before
    # Benign capped at 24,000 (3x)
    # Impact capped at 8,000
    # All rare stages oversampled to 8,000
    print("\nBalancing training sequences...")
    X_train, y_train, yn_train = balance_sequences(
        X_train, y_train, yn_train, min_samples=8000
    )

    # ── Balance VALIDATION sequences ─────────────────────────
    # Balance val too so macro F1 is honest per-stage score
    # Does NOT cause leakage — val rows never used in training
    # min_samples=1000: smaller than train, still representative
    print("\nBalancing validation sequences...")
    X_val, y_val, yn_val = balance_sequences(
        X_val, y_val, yn_val, min_samples=1000
    )

    # ── Datasets and loaders ─────────────────────────────────
    train_dataset = AlertSequenceDataset(
        X_train, y_train, yn_train
    )
    val_dataset = AlertSequenceDataset(
        X_val, y_val, yn_val
    )

    # batch_size=256: sequences are small (5×6), safe
    train_loader = DataLoader(
        train_dataset, batch_size=256,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256,
        shuffle=False, num_workers=0
    )

    # ── Device ────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\nTraining on: {device}")

    # ── Model ─────────────────────────────────────────────────
    # input_size=6  : 6 network flow features per timestep
    # hidden_size=128: 128 units per direction (256 total)
    # num_layers=2  : 2 stacked BiLSTM layers
    # num_stages=7  : 7 MITRE ATT&CK stages
    model = AttackSequenceBiLSTM(
        input_size  = len(FEATURE_COLS),
        hidden_size = 128,
        num_layers  = 2,
        num_stages  = NUM_STAGES,
        dropout     = 0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if device.type == 'cuda':
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"VRAM after model load: {vram:.2f} GB")

    # ── Optimizer ─────────────────────────────────────────────
    # Adam with L2 regularization prevents overfitting
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )

    # Halve lr when macro F1 plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────
    print("\nStarting training...")
    print("=" * 70)

    best_f1          = 0.0
    best_epoch       = 0
    train_losses     = []
    val_f1s          = []
    patience_counter = 0
    early_stop       = 15

    for epoch in range(1, 101):

        loss = train_epoch(
            model, train_loader, optimizer, device
        )
        stage_acc, next_acc, stage_f1 = evaluate(
            model, val_loader, device
        )

        train_losses.append(loss)
        val_f1s.append(stage_f1)
        scheduler.step(stage_f1)

        if epoch % 5 == 0 or epoch == 1:
            if device.type == 'cuda':
                vram = torch.cuda.memory_allocated(0) / 1024**3
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss:.4f} | "
                    f"Stage Acc: {stage_acc:.4f} | "
                    f"Next Acc: {next_acc:.4f} | "
                    f"Stage F1: {stage_f1:.4f} | "
                    f"VRAM: {vram:.2f}GB"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss:.4f} | "
                    f"Stage Acc: {stage_acc:.4f} | "
                    f"Stage F1: {stage_f1:.4f}"
                )

        # Save best model whenever F1 improves
        if stage_f1 > best_f1:
            best_f1    = stage_f1
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                'models/bilstm/best_model.pt'
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"No improvement for {early_stop} epochs")
            break

    print("=" * 70)
    print(f"\nTraining complete!")
    print(f"Best Stage F1: {best_f1:.4f}")
    print(f"Best epoch   : {best_epoch}")

    # ── Training curves ───────────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='steelblue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, color='darkorange')
    plt.title('Validation Stage F1 (Macro)')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/bilstm/training_curves.png', dpi=150)
    print("Curves saved to models/bilstm/training_curves.png")

    # ── Final evaluation on best saved model ──────────────────
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load('models/bilstm/best_model.pt',
                   weights_only=True)
    )
    stage_acc, next_acc, stage_f1 = evaluate(
        model, val_loader, device
    )

    print(f"\nFinal Results on Validation Set:")
    print(f"  Stage Accuracy : {stage_acc:.4f}")
    print(f"  Next Accuracy  : {next_acc:.4f}")
    print(f"  Stage Macro F1 : {stage_f1:.4f}")

    # Save feature cols for inference
    with open('models/bilstm/feature_cols.pkl', 'wb') as f:
        pickle.dump(FEATURE_COLS, f)

    print(f"\nAll files saved to models/bilstm/")
    print("Ready for pipeline integration.")