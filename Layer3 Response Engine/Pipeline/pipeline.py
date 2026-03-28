"""
IMMUNEX - Cybersecurity Dataset Preprocessing Pipeline
=======================================================
Prepares a unified, fully-numeric, normalized dataset from multiple
heterogeneous cybersecurity CSV datasets (CICIDS, NSL-KDD, UNSW-NB15, etc.)
for use in a Gym environment and DQN-based cyber incident response agent.

Author  : IMMUNEX ML Pipeline
Version : 2.0.0 (bug-free)

FIXES IN THIS VERSION:
  1. Duplicate column names removed at load time AND after normalization
     (fixes: AttributeError 'DataFrame has no attribute dtype')
  2. isinstance(series, pd.DataFrame) guard in clean_data() as safety net
  3. All inplace=True removed (deprecated in pandas 3.x)
  4. str(series.dtype) used instead of series.dtype == object
  5. All Unicode arrow characters replaced with ASCII ->
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DATASET_DIR       = "Dataset"   # matches actual folder name (capital D)
OUTPUT_FILE       = "processed_dataset.csv"
MAX_ROWS_PER_FILE = 50_000
CHUNK_SIZE        = 10_000
BALANCE_DATA      = True        # Oversample minority class

COLUMN_ALIASES = {
    "duration": [
        "duration", "dur", "flow_duration",
        "flow duration", "conn_duration"
    ],
    "protocol": [
        "protocol", "protocol_type", "proto",
        "protocol type"
    ],
    "service": [
        "service", "serv", "dst_port",
        "destination_port", "dport", "dest_port",
        "dst port"
    ],
    "src_bytes": [
        "src_bytes", "source_bytes", "sbytes",
        "totlen_fwd_pkts", "total_length_of_fwd_packets",
        "total fwd packets", "fwd_pkts_tot",
        "total_fwd_packets", "totfwdpkts",
        "total length of fwd packets",
        "fwd_header_length", "fwd header length"
    ],
    "dst_bytes": [
        "dst_bytes", "destination_bytes", "dbytes",
        "totlen_bwd_pkts", "total_length_of_bwd_packets",
        "total backward packets", "bwd_pkts_tot",
        "total_backward_packets", "totbwdpkts",
        "total length of bwd packets",
        "bwd_header_length", "bwd header length"
    ],
    "flag": [
        "flag", "flags", "status", "fwd_psh_flags",
        "tcp_flags", "fin_flag_cnt", "syn_flag_cnt",
        "fin flag count", "syn flag count",
        "fwd psh flags", "rst_flag_cnt"
    ],
    "src_packets": [
        # FIX W1: removed aliases that overlapped with src_bytes
        # (total_fwd_packets, fwd_pkts_tot, totfwdpkts are byte counts in CICIDS naming)
        "src_pkts", "spkts",
        "fwd_pkt_len_max", "fwd pkt len max"
    ],
    "dst_packets": [
        # FIX W1: removed aliases that overlapped with dst_bytes
        "dst_pkts", "dpkts",
        "bwd_pkt_len_max", "bwd pkt len max"
    ],
    "src_ip_bytes": [
        "sip_bytes", "sbytes_ip", "src_ip_bytes",
        "flow_byts_s", "flow bytes/s",
        "flow_bytes_s", "byts"
    ],
    "dst_ip_bytes": [
        "dip_bytes", "dbytes_ip", "dst_ip_bytes",
        "flow_pkts_s", "flow packets/s",
        "flow_packets_s", "pkts"
    ],
    "label": [
        "label", "attack_type", "class", "category",
        " label", "attack", "type", "attack_cat",
        "intrusion_type", "traffic_type",
        "flow_label", " Label", "Label",
        "intrusion type", "attack cat"
    ],
}

REQUIRED_COLUMNS = list(COLUMN_ALIASES.keys())

# -----------------------------------------------------------------------------
# STEP 1 - FILE DISCOVERY
# -----------------------------------------------------------------------------

def discover_data_files(root_dir: str) -> list:
    SKIP_KEYWORDS = [
        "features", "list_events", "feature_names",
        "readme", "labels_cleaned", "description"
    ]
    files = []
    for ext in ["*.csv", "*.parquet"]:
        pattern = os.path.join(root_dir, "**", ext)
        all_files = sorted(glob.glob(pattern, recursive=True))
        for f in all_files:
            basename = os.path.basename(f).lower()
            if any(kw in basename for kw in SKIP_KEYWORDS):
                print(f"  [Skip]    '{f}'  -- metadata file")
                continue
            files.append(f)
    print(f"[Discovery] Found {len(files)} data file(s) under '{root_dir}'.")
    return files

# -----------------------------------------------------------------------------
# STEP 2 - SAFE CSV LOADING
# FIX: deduplicate columns immediately after loading
# -----------------------------------------------------------------------------

def load_data_safe(filepath: str,
                   max_rows: int = MAX_ROWS_PER_FILE,
                   chunk_size: int = CHUNK_SIZE):
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".parquet":
        try:
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(filepath)
                chunks = []
                rows_loaded = 0
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    chunk = batch.to_pandas()
                    remaining = max_rows - rows_loaded
                    if remaining <= 0:
                        break
                    chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                    chunks.append(chunk.iloc[:remaining])
                    rows_loaded += len(chunks[-1])
                
                if not chunks:
                    print(f"  [WARNING] '{filepath}' is empty -- skipping.")
                    return None
                    
                df = pd.concat(chunks, ignore_index=True)
                df = df.loc[:, ~df.columns.duplicated()]
                print(f"  [Loaded]  '{filepath}'  ->  {len(df):,} rows x {df.shape[1]} cols")
                return df
            except ImportError:
                df = pd.read_parquet(filepath)
                if len(df) > max_rows:
                    df = df.iloc[:max_rows].copy()
                df = df.loc[:, ~df.columns.duplicated()]
                print(f"  [Loaded]  '{filepath}'  ->  {len(df):,} rows x {df.shape[1]} cols")
                return df
        except Exception as exc:
            print(f"  [ERROR]   Cannot read parquet '{filepath}': {exc}")
            return None

    # Default to CSV handling
    chunks      = []
    rows_loaded = 0
    try:
        reader = pd.read_csv(
            filepath,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
        for chunk in reader:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            # FIX: remove duplicate columns per chunk
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]
            chunks.append(chunk.iloc[:remaining])
            rows_loaded += len(chunks[-1])

        if not chunks:
            print(f"  [WARNING] '{filepath}' is empty -- skipping.")
            return None

        df = pd.concat(chunks, ignore_index=True)

        # FIX: deduplicate again after concat
        df = df.loc[:, ~df.columns.duplicated()]

        print(f"  [Loaded]  '{filepath}'  ->  {len(df):,} rows x {df.shape[1]} cols")
        return df

    except Exception as exc:
        print(f"  [ERROR]   Cannot read '{filepath}': {exc}")
        return None

# -----------------------------------------------------------------------------
# STEP 3 - SCHEMA NORMALIZATION
# FIX: deduplicate after column name normalization
# -----------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[\s\-\/]+", "_", regex=True)
          .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    # FIX: after normalization two columns may become identical name
    df = df.loc[:, ~df.columns.duplicated()]

    alias_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_map[alias.strip().lower()] = canonical

    df = df.rename(columns=alias_map)

    # FIX: deduplicate once more after renaming
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def select_unified_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[available].copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[REQUIRED_COLUMNS]

# -----------------------------------------------------------------------------
# STEP 4 - CLEANING
# FIX 1: deduplicate columns at top of function as final safety net
# FIX 2: isinstance guard so df[col] is always a Series before .dtype
# FIX 3: no inplace=True anywhere
# FIX 4: str(series.dtype) == 'object' instead of series.dtype == object
# -----------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # FIX: final dedup guard
    df = df.loc[:, ~df.columns.duplicated()]

    before = len(df)

    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        series = df[col]

        # FIX: if somehow still a DataFrame (e.g. remaining dups), take first col
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            df = df.drop(columns=[col])
            df[col] = series

        if str(series.dtype) == 'object':
            df[col] = series.fillna("unknown")
        else:
            try:
                df[col] = series.fillna(series.median())
            except Exception:
                df[col] = series.fillna(0)

    df = df.dropna(how="all")

    after = len(df)
    print(f"  [Cleaned] {before - after:,} rows removed  ->  {after:,} rows remain.")
    return df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# STEP 5 - ENCODING & NORMALISATION
# -----------------------------------------------------------------------------

def encode_data(df: pd.DataFrame):
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  [Encode]  '{col}'  ->  {len(le.classes_)} unique class(es)")
    return df, encoders


def normalize_features(df: pd.DataFrame, label_col: str = "label"):
    feature_cols = [c for c in df.columns if c != label_col]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].astype(float))
    print(f"  [Scale]   MinMaxScaler applied to {len(feature_cols)} feature(s).")
    return df, scaler

# -----------------------------------------------------------------------------
# STEP 6 - FEATURE ENGINEERING (128-dim state vector)
# -----------------------------------------------------------------------------

TARGET_DIM = 128

def engineer_features(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    eps = 1e-9

    df["bytes_ratio"]   = df["src_bytes"]   / (df["dst_bytes"]   + eps)
    df["packet_ratio"]  = df["src_packets"] / (df["dst_packets"] + eps)
    df["bytes_per_pkt"] = (df["src_bytes"] + df["dst_bytes"]) / (
                           df["src_packets"] + df["dst_packets"] + eps)

    df["log_src_bytes"]   = np.log1p(df["src_bytes"].clip(lower=0))
    df["log_dst_bytes"]   = np.log1p(df["dst_bytes"].clip(lower=0))
    df["log_src_packets"] = np.log1p(df["src_packets"].clip(lower=0))
    df["log_dst_packets"] = np.log1p(df["dst_packets"].clip(lower=0))
    df["log_duration"]    = np.log1p(df["duration"].clip(lower=0))

    df["bytes_product"] = df["src_bytes"]   * df["dst_bytes"]
    df["pkt_product"]   = df["src_packets"] * df["dst_packets"]

    proto_vals = df["protocol"].value_counts()
    if len(proto_vals) >= 3:
        tcp_code, udp_code, icmp_code = proto_vals.index[:3]
    elif len(proto_vals) == 2:
        tcp_code, udp_code, icmp_code = proto_vals.index[0], proto_vals.index[1], -1
    else:
        tcp_code, udp_code, icmp_code = proto_vals.index[0], -1, -1

    df["is_tcp"]  = (df["protocol"] == tcp_code).astype(np.float32)
    df["is_udp"]  = (df["protocol"] == udp_code).astype(np.float32)
    df["is_icmp"] = (df["protocol"] == icmp_code).astype(np.float32)

    feature_cols = [c for c in df.columns if c != label_col]
    current_dim  = len(feature_cols)
    pad_needed   = TARGET_DIM - current_dim

    if pad_needed < 0:
        feature_cols = feature_cols[:TARGET_DIM]
        df = df[feature_cols + [label_col]]
        print(f"  [Engineer] Trimmed to {TARGET_DIM} features (had {current_dim}).")
    elif pad_needed > 0:
        for i in range(pad_needed):
            df[f"pad_{i}"] = np.float32(0.0)
        print(f"  [Engineer] Added {pad_needed} padding column(s)  ->  "
              f"{TARGET_DIM} features total.")
    else:
        print(f"  [Engineer] Exactly {TARGET_DIM} features -- no padding needed.")

    return df

# -----------------------------------------------------------------------------
# STEP 7 - PER-FILE PROCESSING
# -----------------------------------------------------------------------------

def process_file(filepath: str):
    print(f"\n[Processing] {filepath}")
    df = load_data_safe(filepath)
    if df is None:
        return None
    df = normalize_columns(df)
    df = select_unified_columns(df)
    df = clean_data(df)
    return df

# -----------------------------------------------------------------------------
# STEP 8 - MERGING
# -----------------------------------------------------------------------------

def merge_datasets(dataframes: list) -> pd.DataFrame:
    merged = pd.concat(dataframes, ignore_index=True, sort=False)
    merged = merged[REQUIRED_COLUMNS]
    before = len(merged)
    merged = merged.drop_duplicates()
    print(f"\n[Merge] Combined {len(dataframes)} file(s)  ->  "
          f"{len(merged):,} rows after removing {before - len(merged):,} duplicates.")
    return merged.reset_index(drop=True)

# -----------------------------------------------------------------------------
# STEP 9 - LABEL BINARISATION
# -----------------------------------------------------------------------------

def binarize_labels(df: pd.DataFrame,
                    label_col: str = "label") -> pd.DataFrame:
    normal_strs = {"normal", "benign", "0", "0.0"}
    
    is_normal = df[label_col].astype(str).str.strip().str.lower().isin(normal_strs)
    df[label_col] = (~is_normal).astype(int)
    
    attack_count  = df[label_col].sum()
    normal_count  = len(df) - attack_count
    print(f"[Labels]  Binary  ->  Normal: {normal_count:,}  |  Attack: {attack_count:,}")
    return df

# -----------------------------------------------------------------------------
# STEP 10 - TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

def split_and_save(df: pd.DataFrame,
                   label_col: str    = "label",
                   test_size: float  = 0.20,
                   random_state: int = 42,
                   train_path: str   = "processed_train.csv",
                   test_path: str    = "processed_test.csv",
                   balance_train_data: bool = True):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

    if balance_train_data:
        print("\n[Balance] Oversampling attack class (1) in training data to match normal class (0)")
        normal_df = train_df[train_df[label_col] == 0]
        attack_df = train_df[train_df[label_col] == 1]
        if len(attack_df) > 0 and len(normal_df) > 0:
            attack_upsampled = attack_df.sample(n=len(normal_df), replace=True, random_state=random_state)
            train_df = pd.concat([normal_df, attack_upsampled])
            train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            print(f"  [Balance] New train distribution -> Normal: {(train_df[label_col]==0).sum():,}, Attack: {(train_df[label_col]==1).sum():,}")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    train_mb = os.path.getsize(train_path) / (1024 ** 2)
    test_mb  = os.path.getsize(test_path)  / (1024 ** 2)
    print(f"  [Split]   Train -> '{train_path}'  {len(train_df):,} rows  ({train_mb:.2f} MB)")
    print(f"  [Split]   Test  -> '{test_path}'   {len(test_df):,} rows  ({test_mb:.2f} MB)")

    return train_df, test_df

# -----------------------------------------------------------------------------
# STEP 11 - SAVE FULL DATASET
# -----------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, output_path: str = OUTPUT_FILE) -> None:
    df.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\n[Saved]   '{output_path}'  --  {len(df):,} rows x {df.shape[1]} cols  "
          f"({size_mb:.2f} MB)")

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

def run_pipeline(dataset_dir: str    = DATASET_DIR,
                 output_file: str    = OUTPUT_FILE,
                 binary_labels: bool = True,
                 balance_data: bool  = BALANCE_DATA) -> pd.DataFrame:

    print("=" * 65)
    print("  IMMUNEX  -- Cybersecurity Dataset Preprocessing Pipeline")
    print("=" * 65)

    csv_files = discover_data_files(dataset_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No data files found inside '{dataset_dir}'. "
            "Please ensure the dataset folder exists and contains CSV/Parquet files."
        )

    processed_frames = []
    for filepath in csv_files:
        df = process_file(filepath)
        if df is not None and not df.empty:
            processed_frames.append(df)

    if not processed_frames:
        raise ValueError("All data files were unreadable or empty. Aborting.")

    unified_df = merge_datasets(processed_frames)

    # FIX C2: binarize label strings BEFORE LabelEncoding.
    # LabelEncoder converts strings like "BENIGN" to arbitrary integer codes;
    # if binarize ran after, it would compare against those codes (not "NORMAL"),
    # potentially flipping which class is 0 vs 1.
    if binary_labels:
        unified_df = binarize_labels(unified_df)

    # Now encode the remaining categorical columns (label is already 0/1 int)
    print("\n[Encode] Running global LabelEncoding on merged dataset ...")
    unified_df, encoders = encode_data(unified_df)

    print("\n[Engineer] Building 128-dim feature set ...")
    unified_df = engineer_features(unified_df)

    print("\n[Normalise] Applying MinMaxScaler to merged feature set ...")
    unified_df, scaler = normalize_features(unified_df)

    unified_df = unified_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    for col in unified_df.columns:
        if col == "label":
            unified_df[col] = unified_df[col].astype(np.int8)
        else:
            unified_df[col] = unified_df[col].astype(np.float32)

    save_dataset(unified_df, output_file)

    print("\n[Split] Creating stratified 80/20 train/test split ...")
    split_and_save(
        unified_df,
        label_col    = "label",
        test_size    = 0.20,
        random_state = 42,
        train_path   = "processed_train.csv",
        test_path    = "processed_test.csv",
        balance_train_data = balance_data,
    )

    print("\n[Done] Pipeline completed successfully.")
    print("       The processed dataset is ready for RL / DQN training.")
    print("=" * 65)

    return unified_df

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    final_df = run_pipeline(
        dataset_dir   = DATASET_DIR,
        output_file   = OUTPUT_FILE,
        binary_labels = True,
    )

    print(f"\nFinal dataset shape : {final_df.shape}")
    print(f"Columns             : {list(final_df.columns)}")
    print(f"Label distribution  :\n{final_df['label'].value_counts()}")
    print(f"Data types          :\n{final_df.dtypes}")
