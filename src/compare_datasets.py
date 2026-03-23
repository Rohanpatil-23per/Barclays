import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("DATASET COMPARISON ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# DATASET 1 — New dataset (6 features, purpose-built)
# ─────────────────────────────────────────────────────────────
print("\n--- DATASET 1: gatv2_cicids.csv (6 features) ---")

df_new = pd.read_csv(
    'data/raw/new_dataset/team_datasets/person2_layer2/gatv2_cicids.csv'
)

FEATURES_NEW = [
    'flow_duration', 'syn_flag_count', 'fin_flag_count',
    'rst_flag_count', 'flow_bytes_s', 'flow_packets_s'
]

X_new = df_new[FEATURES_NEW].values
y_new = df_new['is_attack'].values

# Replace inf
X_new = np.where(np.isinf(X_new), 0, X_new)
X_new = np.where(np.isnan(X_new), 0, X_new)

X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
f1_new = f1_score(y_test, rf.predict(X_test))

print(f"  Rows          : {len(df_new):,}")
print(f"  Features      : {len(FEATURES_NEW)}")
print(f"  Attack rows   : {y_new.sum():,} ({y_new.mean()*100:.1f}%)")
print(f"  Benign rows   : {(1-y_new).sum():,} ({(1-y_new).mean()*100:.1f}%)")
print(f"  Random Forest F1 (10 trees): {f1_new:.4f}")
print(f"  Feature overlap between classes:")

for feat in FEATURES_NEW:
    attack_mean = df_new[df_new['is_attack']==1][feat].mean()
    benign_mean = df_new[df_new['is_attack']==0][feat].mean()
    attack_std  = df_new[df_new['is_attack']==1][feat].std()
    benign_std  = df_new[df_new['is_attack']==0][feat].std()
    # Cohen's d — measures separation between classes
    # >0.8 = large separation, <0.2 = small separation
    pooled_std = np.sqrt((attack_std**2 + benign_std**2) / 2)
    cohens_d   = abs(attack_mean - benign_mean) / (pooled_std + 1e-8)
    print(f"    {feat:<25} Cohen's d = {cohens_d:.2f} "
          f"({'VERY EASY' if cohens_d > 2 else 'EASY' if cohens_d > 0.8 else 'HARD'})")

# ─────────────────────────────────────────────────────────────
# DATASET 2 — Old dataset (78 features, raw CICIDS2017)
# ─────────────────────────────────────────────────────────────
print("\n--- DATASET 2: CICIDS2017 parquet (78 features) ---")

import os
folder = 'data/raw/immunex/datasets/cicids2017'
dfs = []
for f in sorted(os.listdir(folder)):
    if f.endswith('.parquet'):
        dfs.append(pd.read_parquet(os.path.join(folder, f)))
df_old = pd.concat(dfs, ignore_index=True)

# Binary label
df_old['is_attack'] = (df_old['Label'] != 'Benign').astype(int)

FEATURES_OLD = [
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s',
    'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count',
    'Packet Length Mean', 'Packet Length Std',
    'Avg Packet Size', 'Packet Length Max',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Active Mean'
]

# Keep only features that exist in the dataframe
FEATURES_OLD = [f for f in FEATURES_OLD if f in df_old.columns]

X_old = df_old[FEATURES_OLD].values
y_old = df_old['is_attack'].values

# Replace inf and nan
X_old = np.where(np.isinf(X_old), 0, X_old)
X_old = np.where(np.isnan(X_old), 0, X_old)

# Sample 200K rows for speed
idx = np.random.choice(len(X_old), size=200000, replace=False)
X_old_sample = X_old[idx]
y_old_sample = y_old[idx]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_old_sample, y_old_sample,
    test_size=0.2, random_state=42, stratify=y_old_sample
)

scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)
X_test2  = scaler2.transform(X_test2)

rf2 = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf2.fit(X_train2, y_train2)
f1_old = f1_score(y_test2, rf2.predict(X_test2))

print(f"  Rows          : {len(df_old):,}")
print(f"  Features      : {len(FEATURES_OLD)}")
print(f"  Attack rows   : {y_old.sum():,} ({y_old.mean()*100:.1f}%)")
print(f"  Benign rows   : {(1-y_old).sum():,} ({(1-y_old).mean()*100:.1f}%)")
print(f"  Random Forest F1 (10 trees): {f1_old:.4f}")
print(f"  Feature overlap between classes:")

for feat in FEATURES_OLD:
    vals = df_old[feat].replace([np.inf, -np.inf], np.nan).dropna()
    attack_mean = df_old[df_old['is_attack']==1][feat].replace(
        [np.inf, -np.inf], np.nan).dropna().mean()
    benign_mean = df_old[df_old['is_attack']==0][feat].replace(
        [np.inf, -np.inf], np.nan).dropna().mean()
    attack_std  = df_old[df_old['is_attack']==1][feat].replace(
        [np.inf, -np.inf], np.nan).dropna().std()
    benign_std  = df_old[df_old['is_attack']==0][feat].replace(
        [np.inf, -np.inf], np.nan).dropna().std()
    pooled_std  = np.sqrt((attack_std**2 + benign_std**2) / 2)
    cohens_d    = abs(attack_mean - benign_mean) / (pooled_std + 1e-8)
    print(f"    {feat:<30} Cohen's d = {cohens_d:.2f} "
          f"({'VERY EASY' if cohens_d > 2 else 'EASY' if cohens_d > 0.8 else 'HARD'})")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  New dataset (6 features):")
print(f"    Random Forest F1  : {f1_new:.4f}")
print(f"    Conclusion        : {'Too easy — model cheats' if f1_new > 0.98 else 'Good difficulty level'}")

print(f"\n  Old dataset (25 features):")
print(f"    Random Forest F1  : {f1_old:.4f}")
print(f"    Conclusion        : {'Too easy' if f1_old > 0.98 else 'Good difficulty — realistic scores expected'}")

print(f"\n  Recommendation:")
if f1_new > 0.98 and f1_old < 0.98:
    print(f"    Use OLD dataset (CICIDS2017 parquet)")
    print(f"    Expected GATv2 F1: 0.85 - 0.93")
elif f1_new > 0.98 and f1_old > 0.98:
    print(f"    Both datasets are too easy for the 6 features")
    print(f"    Need to add harder features or more attack types")
else:
    print(f"    New dataset is fine — proceed with training")