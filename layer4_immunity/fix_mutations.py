"""
Fix mutation normalization — makes mutations same scale as training data
Run this ONCE to regenerate mutated_attacks.csv with correct scale
"""

import pandas as pd
import numpy as np

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

TRAIN_CSV    = r"E:\immunex_p4\person4_layer4\lora_retrain_source.csv"
MUTATION_CSV = r"E:\immunex_p4\layer4_immunity\mutated_attacks.csv"
OUTPUT_CSV   = r"E:\immunex_p4\layer4_immunity\mutated_attacks.csv"

# ── Step 1: Parse training data to get exact feature statistics ───────────────
print("📂 Loading training data to compute feature statistics...")

def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try: lookup[key] = float(val)
            except: lookup[key] = 0.0
    return [lookup.get(f, 0.0) for f in FEATURE_NAMES]

df_train = pd.read_csv(TRAIN_CSV)
X_train  = np.array([parse_text(t) for t in df_train["text"]])

# Compute per-feature mean and std from training data
feat_mean = X_train.mean(axis=0)
feat_std  = X_train.std(axis=0) + 1e-8  # avoid division by zero

print(f"   ✅ Training stats computed from {len(X_train):,} samples")
print(f"   Training scale: min={X_train.min():.3f} max={X_train.max():.3f}")

# ── Step 2: Load mutations ─────────────────────────────────────────────────────
print("\n📂 Loading mutations...")
df_mut   = pd.read_csv(MUTATION_CSV)
X_mut    = df_mut[FEATURE_NAMES].values.astype(np.float64)
print(f"   Raw mutation scale: min={X_mut.min():.3f} max={X_mut.max():.3f}")

# ── Step 3: Normalize mutations using training statistics ──────────────────────
print("\n🔄 Normalizing mutations to training data scale...")

# Per-feature z-score normalization using TRAINING mean/std
X_norm = (X_mut - feat_mean) / feat_std

# Clip to same range as training data (±5 std covers all real traffic)
X_norm = np.clip(X_norm, -5, 5)

print(f"   ✅ Normalized scale: min={X_norm.min():.3f} max={X_norm.max():.3f}")

# Verify per feature
print("\n   Per feature check (first 5):")
for i, f in enumerate(FEATURE_NAMES[:5]):
    t_min = X_train[:, i].min()
    t_max = X_train[:, i].max()
    m_min = X_norm[:, i].min()
    m_max = X_norm[:, i].max()
    print(f"   {f}:")
    print(f"     train: {t_min:.3f} to {t_max:.3f}")
    print(f"     mut:   {m_min:.3f} to {m_max:.3f}  "
          f"{'✅' if abs(m_min) < 6 and abs(m_max) < 6 else '❌'}")

# ── Step 4: Save fixed mutations ───────────────────────────────────────────────
print("\n💾 Saving normalized mutations...")
df_fixed = pd.DataFrame(X_norm, columns=FEATURE_NAMES)

# Keep metadata columns
for col in ["mutation_label", "attack_cat", "mutation_strategy",
            "mutation_variant"]:
    if col in df_mut.columns:
        df_fixed[col] = df_mut[col].values

df_fixed.to_csv(OUTPUT_CSV, index=False)
print(f"   ✅ Saved {len(df_fixed):,} normalized mutations to:")
print(f"   {OUTPUT_CSV}")
print(f"\n🎉 Normalization complete! Now run ewc.py again.")
