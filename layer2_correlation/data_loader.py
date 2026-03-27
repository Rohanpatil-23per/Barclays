import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle

# ─────────────────────────────────────────────────────────────
# LABEL MAPPING — maps raw labels to MITRE ATT&CK stages
# ─────────────────────────────────────────────────────────────

LABEL_TO_MITRE = {
    'Benign'              : 'Benign',
    'Normal'              : 'Benign',
    'PortScan'            : 'Reconnaissance',
    'Reconnaissance'      : 'Reconnaissance',
    'Fuzzers'             : 'Reconnaissance',
    'Analysis'            : 'Reconnaissance',
    'FTP-Patator'         : 'Initial_Access',
    'SSH-Patator'         : 'Initial_Access',
    'Exploits'            : 'Initial_Access',
    'Shellcode'           : 'Initial_Access',
    'Bot'                 : 'Execution',
    'Backdoor'            : 'Execution',
    'Worms'               : 'Execution',
    'DDoS'                : 'Impact',
    'DoS Hulk'            : 'Impact',
    'DoS GoldenEye'       : 'Impact',
    'DoS slowloris'       : 'Impact',
    'DoS Slowhttptest'    : 'Impact',
    'Generic'             : 'Impact',
    'DoS'                 : 'Impact',
    'Infiltration'        : 'Exfiltration',
    'Heartbleed'          : 'Exploitation',
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

# ─────────────────────────────────────────────────────────────
# FUNCTION 1 — Fix Web Attack encoding issues
# Detects web attack variants by keyword instead of exact match
# ─────────────────────────────────────────────────────────────

def fix_labels(label):
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
# FUNCTION 2 — Load gatv2_cicids.csv
# Purpose-built for GATv2 with 6 key network flow features
# ─────────────────────────────────────────────────────────────

def load_cicids(filepath):
    print(f"Loading CICIDS dataset...")
    df = pd.read_csv(filepath)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.columns.tolist()}")

    # Fix web attack label encoding issues
    df['label'] = df['label'].apply(fix_labels)

    # Standardize label column name
    df = df.rename(columns={'label': 'Label'})

    print(f"  Nulls   : {df.isnull().sum().sum()}")
    print(f"  Label distribution:")
    print(df['Label'].value_counts())
    return df

# ─────────────────────────────────────────────────────────────
# FUNCTION 3 — Load gatv2_unsw.csv
# Supplementary dataset from different network environment
# Helps model generalize beyond CICIDS traffic patterns
# ─────────────────────────────────────────────────────────────

UNSW_TO_CICIDS = {
    'dur'    : 'flow_duration',
    'sbytes' : 'flow_bytes_s',
    'dbytes' : 'flow_packets_s',
}

def load_unsw(filepath):
    print(f"\nLoading UNSW dataset...")
    df = pd.read_csv(filepath)

    cols = list(UNSW_TO_CICIDS.keys()) + ['attack_cat', 'label']
    df   = df[cols]

    df = df.rename(columns=UNSW_TO_CICIDS)
    df = df.rename(columns={'attack_cat': 'Label'})

    # UNSW doesn't have flag counts — fill with 0
    for col in ['syn_flag_count', 'fin_flag_count', 'rst_flag_count']:
        df[col] = 0

    df = df.drop(columns=['label'])

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.columns.tolist()}")
    print(f"  Nulls   : {df.isnull().sum().sum()}")
    return df

# ─────────────────────────────────────────────────────────────
# FUNCTION 4 — Add MITRE labels to any dataframe
# Maps raw label strings to MITRE stage names and numeric IDs
# ─────────────────────────────────────────────────────────────

def add_mitre_labels(df):
    df['mitre_stage'] = df['Label'].map(LABEL_TO_MITRE)

    unmapped = df['mitre_stage'].isnull().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} unmapped labels:")
        print(f"  {df[df['mitre_stage'].isnull()]['Label'].unique()}")
        df['mitre_stage'] = df['mitre_stage'].fillna('Benign')

    df['mitre_id']  = df['mitre_stage'].map(MITRE_TO_ID)
    df['is_attack'] = (df['mitre_stage'] != 'Benign').astype(int)
    return df

# ─────────────────────────────────────────────────────────────
# FUNCTION 5 — Scale features to mean=0, std=1
# Neural networks train much better on normalized data
# We do NOT balance classes here anymore —
# balancing happens in graph_builder.py AFTER the split
# to prevent duplicate rows leaking across train/val sets
# ─────────────────────────────────────────────────────────────

def scale_features(df, feature_cols):
    print("\nScaling features...")

    # Replace inf values from flow rate calculations
    df[feature_cols] = df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    df[feature_cols] = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"  Scaled {len(feature_cols)} features")
    print(f"  Mean (should be ~0): {df[feature_cols].mean().mean():.4f}")
    print(f"  Std  (should be ~1): {df[feature_cols].std().mean():.4f}")

    return df, scaler

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CICIDS_FILE = 'data/raw/new_dataset/team_datasets/person2_layer2/gatv2_cicids.csv'
    UNSW_FILE   = 'data/raw/new_dataset/team_datasets/person2_layer2/gatv2_unsw.csv'

    FEATURE_COLS = [
        'flow_duration',
        'syn_flag_count',
        'fin_flag_count',
        'rst_flag_count',
        'flow_bytes_s',
        'flow_packets_s'
    ]

    # ── PIPELINE 1: CICIDS only ────────────────────────────────
    print("=" * 55)
    print("PIPELINE 1: CICIDS (6 features for GATv2)")
    print("=" * 55)

    cicids = load_cicids(CICIDS_FILE)
    cicids = add_mitre_labels(cicids)

    print("\nMITRE Distribution:")
    print(cicids['mitre_stage'].value_counts())

    # No balancing here — balancing happens in graph_builder.py
    # after the train/val split to prevent data leakage
    cicids_final, scaler_cicids = scale_features(
        cicids, FEATURE_COLS
    )

    cicids_final.to_parquet(
        'data/processed/cicids_final.parquet', index=False
    )
    with open('data/processed/scaler_cicids.pkl', 'wb') as f:
        pickle.dump(scaler_cicids, f)
    with open('data/processed/feature_cols_cicids.pkl', 'wb') as f:
        pickle.dump(FEATURE_COLS, f)

    print(f"\nCICIDS pipeline done!")
    print(f"Shape  : {cicids_final.shape}")
    print(f"Features: {FEATURE_COLS}")

    # ── PIPELINE 2: CICIDS + UNSW combined ────────────────────
    print("\n" + "=" * 55)
    print("PIPELINE 2: CICIDS + UNSW (combined for validation)")
    print("=" * 55)

    unsw = load_unsw(UNSW_FILE)
    unsw = add_mitre_labels(unsw)

    cicids_subset = cicids[FEATURE_COLS + ['Label', 'mitre_stage',
                                            'mitre_id', 'is_attack']]
    unsw_subset   = unsw[FEATURE_COLS + ['Label', 'mitre_stage',
                                          'mitre_id', 'is_attack']]

    merged = pd.concat(
        [cicids_subset, unsw_subset], ignore_index=True
    )
    print(f"\nMerged total: {len(merged):,} rows")
    print("MITRE Distribution:")
    print(merged['mitre_stage'].value_counts())

    merged_final, scaler_merged = scale_features(
        merged, FEATURE_COLS
    )

    merged_final.to_parquet(
        'data/processed/final_dataset.parquet', index=False
    )
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler_merged, f)

    print(f"\nCombined pipeline done!")
    print(f"Shape: {merged_final.shape}")
    print(f"\nAll data saved to data/processed/")
    print("Ready for graph building.")