import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
warnings.filterwarnings('ignore')

# ================================================================
# DEVICE SETUP
# ================================================================
print("=" * 60)
print("IMMUNEX — LSTM-HMM Final Training")
print("=" * 60)

print(f"\n  PyTorch version:           {torch.__version__}")
print(f"  PyTorch built with CUDA:   {torch.version.cuda}")
print(f"  CUDA available:            {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"\n  [GPU] Device:   {props.name}")
    print(f"  [GPU] VRAM:     {props.total_memory / 1e9:.1f} GB")
    print(f"  [GPU] CUDA:     {torch.version.cuda}")
else:
    print(f"\n  [CPU] Running on CPU — training will be slow")

# ================================================================
# CONFIGURATION — your file paths kept exactly as they were
# ================================================================
CONFIG = {
    "cicids_path":    "F:/Barclays/immunex/Aditya Dataset/team_datasets/person5_layer5/hmm_sequences_cicids.csv",
    "nslkdd_path":    "F:/Barclays/immunex/Aditya Dataset/team_datasets/person5_layer5/hmm_nslkdd.csv",
    "network_path":   "F:/Barclays/immunex/Aditya Dataset/team_datasets/person5_layer5/train_test_network.csv",
    "n_states":       5,
    "n_observations": 10,
    "seq_length":     20,
    "embed_dim":      32,
    "lstm_hidden":    128,
    "lstm_layers":    2,
    "dropout":        0.3,
    "batch_size":     512 if torch.cuda.is_available() else 256,
    "epochs":         40,
    "learning_rate":  0.001,
    "n_chains":       60000,
    "output_path":    "immunex_lstm_final.pt",
}

STATE_NAMES = ['Recon', 'Init Access', 'Priv Esc', 'Lateral Mv', 'Exfiltration']
OBS_NAMES   = [
    'port_scan', 'dns_query', 'phishing_click', 'login_fail',
    'login_success', 'priv_escalation', 'lateral_movement',
    'file_access', 'large_upload', 'zip_creation'
]

# ================================================================
# STEP 1 — LOAD DATA
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading datasets")
print("=" * 60)

def load_csv(path, name):
    df = pd.read_csv(path)[['hmm_state', 'observation']].dropna()
    df = df[df['hmm_state'].between(0, CONFIG['n_states'] - 1)]
    df = df[df['observation'].between(0, CONFIG['n_observations'] - 1)]
    df['hmm_state']   = df['hmm_state'].astype(int)
    df['observation'] = df['observation'].astype(int)
    print(f"  {name}: {len(df):,} rows | states: {sorted(df['hmm_state'].unique())}")
    return df

df_all = pd.concat([
    load_csv(CONFIG['cicids_path'],  'CICIDS-2017'),
    load_csv(CONFIG['nslkdd_path'],  'NSL-KDD'),
    load_csv(CONFIG['network_path'], 'Network'),
], ignore_index=True)

print(f"\n  Combined: {len(df_all):,} events")
print("\n  Raw state distribution:")
for sid in range(5):
    count = (df_all['hmm_state'] == sid).sum()
    pct   = count / len(df_all) * 100
    flag  = " <- RARE (will be fixed by chain generation)" if pct < 2.0 else ""
    print(f"    {STATE_NAMES[sid]:<20}: {count:>8,}  ({pct:.1f}%){flag}")

# ================================================================
# STEP 2 — LEARN EMISSION PROBABILITIES FROM REAL DATA
#
# WHY WE DO THIS INSTEAD OF SLIDING WINDOW:
#
# Your CSV rows are individual events from DIFFERENT attacks.
# A sliding window across them creates sequences like:
#   [Recon_event, PrivEsc_event, Exfil_event, Recon_event...]
# — these events are from 4 completely unrelated attacks.
# The LSTM sees random noise and cannot learn anything.
#
# Instead we extract: "In state X, which observations appear?"
# This is the real information in your CSVs.
# Then we generate proper sequential chains that the LSTM CAN learn from.
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Learning emission probabilities from real data")
print("=" * 60)

emission_probs = {}
print(f"\n  {'State':<20} Top observations (from your real data)")
print("  " + "-" * 60)

for state in range(5):
    rows = df_all[df_all['hmm_state'] == state]['observation']
    if len(rows) == 0:
        # Fallback: use reasonable default if state missing
        defaults = {0:[0], 1:[1,2,3], 2:[3,4,5], 3:[5,6,7], 4:[7,8,9]}
        obs_list = defaults.get(state, [state])
        emission_probs[state] = {o: 1.0/len(obs_list) for o in obs_list}
        print(f"  {STATE_NAMES[state]:<20} [using default — state absent in data]")
        continue

    counts = rows.value_counts()
    probs  = (counts / counts.sum()).to_dict()
    emission_probs[state] = probs

    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    obs_str = "  ".join([f"{OBS_NAMES[o]}({p:.0%})" for o, p in top3])
    print(f"  {STATE_NAMES[state]:<20} {obs_str}")

# ================================================================
# STEP 3 — GENERATE SEQUENTIAL ATTACK CHAINS
#
# Each chain = one complete attack from start to finish.
# The chain follows the real kill chain order.
# The observations within each stage are sampled from your real data.
#
# Example chain generated:
#   Stage 0 (Recon):      [port_scan, port_scan, port_scan, dns_query]
#   Stage 1 (Init Acc):   [dns_query, login_fail, login_fail, phishing_click]
#   Stage 2 (Priv Esc):   [login_fail, login_fail, login_success, priv_esc]
#   Stage 3 (Lateral Mv): [priv_esc, lateral_movement, lateral_movement]
#   Stage 4 (Exfil):      [file_access, large_upload, zip_creation]
#
# Full sequence: [0,0,0,1,1,3,3,2,3,3,4,5,5,4,5,6,6,7,8,9]
#
# NOW the LSTM sees proper temporal patterns and can learn:
#   "port_scan repeated → probably still Recon"
#   "login_fail x3 then login_success → transitioning to Priv Esc"
#   "file_access then large_upload → Exfiltration happening"
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Generating sequential attack chains")
print("=" * 60)

# All realistic attack path variations
ATTACK_PATHS = [
    # Full kill chain (most important — appears most often)
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    # Skip lateral movement (direct escalation to exfil)
    [0, 1, 2, 4],
    [0, 1, 2, 4],
    # Extended recon before access
    [0, 0, 1, 2, 3, 4],
    # Slow methodical attacker
    [0, 1, 1, 2, 2, 3, 3, 4],
    # Skip initial access (direct exploitation)
    [0, 2, 3, 4],
    # Partial attack (caught early at recon)
    [0, 0, 1],
    # Partial attack (caught at privilege escalation)
    [0, 1, 2, 3],
    # Rapid attack (minimal dwell time)
    [0, 1, 2, 4],
    # Insider threat (starts at privilege escalation)
    [2, 3, 4],
]

def sample_obs_for_state(state, n):
    """Sample n observations from a state using real emission probabilities."""
    obs_ids = list(emission_probs[state].keys())
    probs   = np.array(list(emission_probs[state].values()), dtype=float)
    probs  /= probs.sum()
    return np.random.choice(obs_ids, size=n, p=probs).tolist()

def generate_chain(path):
    """Generate one complete attack chain following the given state path."""
    obs_seq   = []
    state_seq = []
    for stage_idx, state in enumerate(path):
        # More events in middle stages, fewer at start/end
        # This mirrors real attack behavior
        if state == 0:   n = np.random.randint(3, 10)   # Recon: 3-9 events
        elif state == 1: n = np.random.randint(2, 7)    # Initial access: 2-6
        elif state == 2: n = np.random.randint(3, 9)    # Priv esc: 3-8
        elif state == 3: n = np.random.randint(3, 10)   # Lateral: 3-9
        else:            n = np.random.randint(2, 7)    # Exfil: 2-6

        obs_seq.extend(sample_obs_for_state(state, n))
        state_seq.extend([state] * n)
    return obs_seq, state_seq

np.random.seed(42)
N = CONFIG['n_chains']
obs_chains, state_chains = [], []

for i in range(N):
    path = ATTACK_PATHS[i % len(ATTACK_PATHS)]
    obs, states = generate_chain(path)
    obs_chains.append(obs)
    state_chains.append(states)

print(f"  Generated {N:,} attack chains")
avg_len = np.mean([len(c) for c in obs_chains])
print(f"  Average chain length: {avg_len:.1f} events")

# Show example chain
ex_obs, ex_st = obs_chains[0], state_chains[0]
print(f"\n  Example chain:")
for i in range(min(20, len(ex_obs))):
    print(f"    event {i+1:>2}: {OBS_NAMES[ex_obs[i]]:<20} [stage: {STATE_NAMES[ex_st[i]]}]")

# Verify all 5 states are covered
chain_states = [s for chain in state_chains for s in chain]
print(f"\n  State distribution in generated chains (should be balanced):")
c = Counter(chain_states)
for sid in range(5):
    count = c[sid]
    pct   = count / len(chain_states) * 100
    bar   = "█" * int(pct / 2)
    print(f"    {STATE_NAMES[sid]:<20}: {count:>8,}  ({pct:.1f}%)  {bar}")

# ================================================================
# STEP 4 — BUILD PYTORCH DATASET
#
# Sliding window is applied WITHIN each chain only.
# This preserves the temporal ordering of events.
#
# For chain: [0, 0, 1, 3, 3, 4, 5]
#   t=1: X=[0]           → predict state=Recon,   obs=0
#   t=2: X=[0, 0]        → predict state=Recon,   obs=1
#   t=3: X=[0, 0, 1]     → predict state=PrivEsc, obs=3
#   t=4: X=[0, 0, 1, 3]  → predict state=PrivEsc, obs=3
#   ...
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: Building LSTM dataset from chains")
print("=" * 60)

SEQ_LEN = CONFIG['seq_length']

class ChainDataset(Dataset):
    def __init__(self, obs_chains, state_chains):
        self.X, self.y_obs, self.y_state = [], [], []
        max_len = SEQ_LEN - 1

        for obs_seq, state_seq in zip(obs_chains, state_chains):
            for t in range(1, len(obs_seq)):
                # Get history up to position t (right-padded from left)
                history = obs_seq[max(0, t - max_len): t]
                padded  = np.zeros(max_len, dtype=np.int64)
                padded[max_len - len(history):] = history

                self.X.append(padded)
                self.y_obs.append(int(obs_seq[t]))
                self.y_state.append(int(state_seq[t]))

        self.X       = np.array(self.X,       dtype=np.int64)
        self.y_obs   = np.array(self.y_obs,   dtype=np.int64)
        self.y_state = np.array(self.y_state, dtype=np.int64)

    def __len__(self): return len(self.y_obs)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx],       dtype=torch.long),
            torch.tensor(self.y_obs[idx],   dtype=torch.long),
            torch.tensor(self.y_state[idx], dtype=torch.long),
        )

# Split chains into train/test (not individual samples)
split     = int(0.8 * N)
tr_ds     = ChainDataset(obs_chains[:split],  state_chains[:split])
te_ds     = ChainDataset(obs_chains[split:],  state_chains[split:])

print(f"  Train samples: {len(tr_ds):,}")
print(f"  Test  samples: {len(te_ds):,}")

BATCH = CONFIG['batch_size']
tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0,
                       pin_memory=torch.cuda.is_available())
te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                       pin_memory=torch.cuda.is_available())

# ================================================================
# STEP 5 — MODEL DEFINITION
#
# Architecture:
#   Embedding(10 obs → 32 dims)
#   → LSTM(32 → 128, 2 layers)
#   → two output heads:
#       obs_head:   predict next observation (10 classes)
#       state_head: predict current attacker stage (5 classes)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: Building LSTM model")
print("=" * 60)

class LSTMAttackerPredictor(nn.Module):
    def __init__(self, n_obs, n_states, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(n_obs + 1, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout    = nn.Dropout(dropout)
        self.obs_head   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_obs)
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_states)
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, _   = self.lstm(embedded)
        last     = self.dropout(out[:, -1, :])
        return self.obs_head(last), self.state_head(last)


model = LSTMAttackerPredictor(
    n_obs      = CONFIG['n_observations'],
    n_states   = CONFIG['n_states'],
    embed_dim  = CONFIG['embed_dim'],
    hidden_dim = CONFIG['lstm_hidden'],
    n_layers   = CONFIG['lstm_layers'],
    dropout    = CONFIG['dropout'],
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {total_params:,}")
print(f"  Hidden dim: {CONFIG['lstm_hidden']}")
print(f"  Embed dim:  {CONFIG['embed_dim']}")
print(f"  On device:  {next(model.parameters()).device}")
if torch.cuda.is_available():
    print(f"  VRAM used:  {torch.cuda.memory_allocated(0)/1e6:.1f} MB")

# ================================================================
# STEP 6 — TRAINING LOOP
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)

criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(),
                               lr=CONFIG['learning_rate'],
                               weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

best_macro = 0.0
best_ep    = 0
no_impr    = 0
EPOCHS     = CONFIG['epochs']
eta = "~10-20 min" if torch.cuda.is_available() else "~2-3 hrs"

print(f"\n  Epochs: {EPOCHS} | Batch: {BATCH} | ETA: {eta}")
print(f"\n  {'Ep':>3} | {'T-Loss':>7} | {'V-Loss':>7} | {'Acc':>6} | {'MacroF1':>8} | {'LR':>8} | VRAM")
print("  " + "-" * 65)

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    t_loss = 0.0
    for X, y_obs, y_st in tr_loader:
        X, y_obs, y_st = (X.to(device, non_blocking=True),
                          y_obs.to(device, non_blocking=True),
                          y_st.to(device, non_blocking=True))
        optimizer.zero_grad()
        o_log, s_log = model(X)
        loss = criterion(o_log, y_obs) + criterion(s_log, y_st)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()
    t_loss /= len(tr_loader)

    # ── Validate ──
    model.eval()
    v_loss, pred, true = 0.0, [], []
    with torch.no_grad():
        for X, y_obs, y_st in te_loader:
            X, y_obs, y_st = (X.to(device, non_blocking=True),
                              y_obs.to(device, non_blocking=True),
                              y_st.to(device, non_blocking=True))
            o_log, s_log = model(X)
            v_loss += (criterion(o_log, y_obs) + criterion(s_log, y_st)).item()
            pred.extend(s_log.argmax(1).cpu().numpy())
            true.extend(y_st.cpu().numpy())
    v_loss   /= len(te_loader)
    acc       = accuracy_score(true, pred)
    macro_f1  = f1_score(true, pred, average='macro', zero_division=0)
    lr        = optimizer.param_groups[0]['lr']
    scheduler.step(macro_f1)

    vram = f"{torch.cuda.memory_allocated(0)/1e6:.0f}MB" if torch.cuda.is_available() else ""
    print(f"  {epoch:>3} | {t_loss:>7.4f} | {v_loss:>7.4f} | {acc:>6.4f} | "
          f"{macro_f1:>8.4f} | {lr:>8.6f} | {vram}")

    if macro_f1 > best_macro:
        best_macro = macro_f1
        best_ep    = epoch
        no_impr    = 0
        torch.save({
            'epoch':            epoch,
            'model_state_dict': model.state_dict(),
            'val_state_acc':    acc,
            'macro_f1':         macro_f1,
            'trained_on':       str(device),
            'config':           CONFIG,
        }, CONFIG['output_path'])
    else:
        no_impr += 1
        if no_impr >= 8:
            print(f"\n  Early stopping at epoch {epoch} — no improvement for 8 epochs")
            break

print(f"\n  Training complete!")
print(f"  Best Macro F1: {best_macro:.4f} at epoch {best_ep}")
print(f"  Saved: {CONFIG['output_path']}")

# ================================================================
# STEP 7 — FINAL EVALUATION REPORT
# ================================================================
print("\n" + "=" * 60)
print("STEP 7: Final evaluation")
print("=" * 60)

ck = torch.load(CONFIG['output_path'], map_location=device)
model.load_state_dict(ck['model_state_dict'])
model.eval()

pred, true = [], []
with torch.no_grad():
    for X, _, y_st in te_loader:
        X = X.to(device)
        _, s_log = model(X)
        pred.extend(s_log.argmax(1).cpu().numpy())
        true.extend(y_st.numpy())

pred = np.array(pred)
true = np.array(true)

print("\n  ATTACKER STAGE PREDICTION — all 5 stages should show >0 recall:")
print()
print(classification_report(true, pred, target_names=STATE_NAMES, zero_division=0))

final_acc   = accuracy_score(true, pred)
final_macro = f1_score(true, pred, average='macro', zero_division=0)
print(f"  Overall accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"  Macro F1:         {final_macro:.4f}")
print(f"  Trained on:       {ck['trained_on']}")

# ================================================================
# STEP 8 — INFERENCE FUNCTION
# This is what Layer 5 / FastAPI will call at runtime
# ================================================================
print("\n" + "=" * 60)
print("STEP 8: Inference test on real attack scenarios")
print("=" * 60)

def predict_attack(observation_history: list) -> dict:
    """
    Given a list of recent observation integers (0-9),
    predict the attacker's current stage and next move.

    Called by Layer 5 pipeline and FastAPI /predict endpoint.
    """
    max_len = SEQ_LEN - 1
    padded  = np.zeros(max_len, dtype=np.int64)
    hist    = np.array(observation_history[-max_len:])
    padded[max_len - len(hist):] = hist

    with torch.no_grad():
        x = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
        o_log, s_log = model(x)

    sp = torch.softmax(s_log, dim=1).squeeze().cpu().numpy()
    op = torch.softmax(o_log, dim=1).squeeze().cpu().numpy()

    return {
        "current_stage":    STATE_NAMES[sp.argmax()],
        "stage_confidence": f"{sp.max()*100:.1f}%",
        "stage_probs":      {STATE_NAMES[i]: round(float(sp[i]), 3) for i in range(5)},
        "predicted_next":   OBS_NAMES[op.argmax()],
        "next_confidence":  f"{op.max()*100:.1f}%",
        "next_obs_probs":   {OBS_NAMES[i]: round(float(op[i]), 3) for i in range(10)},
    }

# 6 real attack scenarios covering all 5 stages
SCENARIOS = [
    {
        "name":   "Port scan → brute force → privilege escalation",
        "seq":    [0, 0, 0, 1, 3, 3, 3, 4, 5],
        "expect": "Priv Esc",
    },
    {
        "name":   "Pure reconnaissance (port scanning only)",
        "seq":    [0, 0, 0, 0, 0, 0, 1, 0],
        "expect": "Recon",
    },
    {
        "name":   "Phishing click → initial access",
        "seq":    [1, 1, 2, 3, 3, 4],
        "expect": "Init Access",
    },
    {
        "name":   "Lateral movement across network",
        "seq":    [5, 5, 6, 6, 7, 6, 6],
        "expect": "Lateral Mv",
    },
    {
        "name":   "Data exfiltration (end stage)",
        "seq":    [6, 7, 7, 8, 8, 8, 9],
        "expect": "Exfiltration",
    },
    {
        "name":   "Full kill chain observed",
        "seq":    [0, 0, 1, 3, 3, 4, 5, 6, 7, 8],
        "expect": "Lateral Mv",
    },
]

correct = 0
print()
for s in SCENARIOS:
    result = predict_attack(s['seq'])
    ok     = result['current_stage'] == s['expect']
    correct += int(ok)
    tick   = "PASS" if ok else "FAIL"

    print(f"  [{tick}] {s['name']}")
    print(f"         Input:    {[OBS_NAMES[o] for o in s['seq']]}")
    print(f"         Expected: {s['expect']:<15}  Got: {result['current_stage']} "
          f"({result['stage_confidence']})")
    print(f"         Next predicted action: {result['predicted_next']} "
          f"({result['next_confidence']})")
    print(f"         Stage breakdown:", end=" ")
    for stage, prob in result['stage_probs'].items():
        if prob > 0.05:
            print(f"{stage}={prob:.2f}", end="  ")
    print("\n")

print(f"  Scenario score: {correct}/{len(SCENARIOS)}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"""
  Output file:  {CONFIG['output_path']}
  Accuracy:     {final_acc*100:.2f}%
  Macro F1:     {final_macro:.4f}
  Best epoch:   {best_ep}
  Trained on:   {ck['trained_on']}

  FILES TO HAND TO PARTH FOR INTEGRATION:
    immunex_hmm.pkl          (unchanged from original training)
    {CONFIG['output_path']}  (use this, NOT any previous .pt file)

  FOR THE BARCLAYS DEMO:
    All 5 attacker stages should now show non-zero recall.
    Target Macro F1 > 0.70 is good for demo day.
    If any stage still shows 0.00, paste the report here.
""")
print("=" * 60)

    CORRECT_SCENARIOS = [
        {
            "name":   "Reconnaissance",
            "seq":    [0, 0, 0, 0, 0, 0, 0],        # obs 0 = port_scan → state 0
            "expect": "Recon",
        },
        {
            "name":   "Initial Access",
            "seq":    [0, 0, 1, 1, 1, 1],            # obs 1 = dns_query → state 1
            "expect": "Init Access",
        },
        {
            "name":   "Privilege Escalation",
            "seq":    [0, 1, 3, 3, 3, 4, 5],         # obs 3,4,5 → state 2
            "expect": "Priv Esc",
        },
        {
            "name":   "Lateral Movement",
            "seq":    [0, 1, 3, 4, 5, 5, 5, 5],      # obs 5 sustained → state 3
            "expect": "Lateral Mv",
        },
        {
            "name":   "Exfiltration",
            "seq":    [0, 1, 3, 4, 5, 5, 6, 7, 8, 9], # obs 6,7,8,9 → state 4
            "expect": "Exfiltration",
        },
    ]

    correct = 0
    for s in CORRECT_SCENARIOS:
        result = predict_attack(s['seq'])
        ok     = result['current_stage'] == s['expect']
        correct += int(ok)
        tick   = "PASS" if ok else "FAIL"
        print(f"  [{tick}] {s['name']}")
        print(f"         Expected: {s['expect']:<15} Got: {result['current_stage']} ({result['stage_confidence']})")
        print(f"         Next: {result['predicted_next']} ({result['next_confidence']})\n")

    print(f"  Score: {correct}/{len(CORRECT_SCENARIOS)}")