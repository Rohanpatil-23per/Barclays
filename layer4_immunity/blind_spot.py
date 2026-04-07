"""
IMMUNEX - Layer 4: Blind Spot Detection
Finds attacks that the trained model misses
Input:  blind_spot_candidates.csv (UNSW-NB15 attacks)
        trained model from lora_retrain.py
Output: blind_spots.csv (attacks model missed)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = BASE_DIR
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR, "lora_model.pt")
BLIND_CSV  = os.path.join(DATA_DIR,  "blind_spot_candidates.csv")
OUTPUT_CSV = os.path.join(BASE_DIR,  "blind_spots.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# ─── Same 25 features as lora_retrain.py ──────────────────────────────────────
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

# ─── Mapping UNSW-NB15 columns → our 25 feature names ─────────────────────────
# UNSW-NB15 uses different column names than our training data
# We map what we can, rest defaults to 0
UNSW_MAPPING = {
    "dur":     "flow_duration",
    "spkts":   "total_fwd_packets",
    "dpkts":   "total_backward_packets",
    "sload":   "flow_bytes/s",
    "rate":    "flow_packets/s",
    "smean":   "fwd_packet_length_mean",
    "dmean":   "bwd_packet_length_mean",
    "sinpkt":  "flow_iat_mean",
    "dinpkt":  "fwd_iat_mean",
    "sjit":    "bwd_iat_mean",
    "synack":  "syn_flag_count",
    "ackdat":  "ack_flag_count",
    "tcprtt":  "fin_flag_count",
    "sbytes":  "packet_length_mean",
    "dbytes":  "packet_length_std",
    "swin":    "init_fwd_win_bytes",
    "dwin":    "init_bwd_win_bytes",
    "ct_srv_src":  "active_mean",
    "ct_srv_dst":  "idle_mean",
    "trans_depth": "down/up_ratio",
    "response_body_len": "avg_packet_size",
}

# ─── Same model architecture as lora_retrain.py ───────────────────────────────
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
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))

# ─── Load trained model ────────────────────────────────────────────────────────
def load_model(device):
    print("📂 Loading trained model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    rank = checkpoint.get("lora_rank", 8)
    model = IMMUNEXLayer4(input_dim=25, rank=rank).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"   ✅ Model loaded | Training accuracy: {checkpoint['accuracy']:.2f}%")
    return model

# ─── Convert UNSW-NB15 row → our 25 features ──────────────────────────────────
def convert_unsw_row(row):
    """
    UNSW-NB15 has 45 columns with different names
    We map them to our 25 feature names
    Unmapped features default to 0.0
    """
    result = {f: 0.0 for f in FEATURE_NAMES}
    for unsw_col, our_col in UNSW_MAPPING.items():
        if unsw_col in row.index:
            try:
                result[our_col] = float(row[unsw_col])
            except:
                result[our_col] = 0.0
    return np.array([result[f] for f in FEATURE_NAMES], dtype=np.float32)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4: BLIND SPOT DETECTION")
    print("="*60)

    # Load model
    model = load_model(device)

    # Load blind spot candidates
    print("\n📂 Loading blind spot candidates...")
    df = pd.read_csv(BLIND_CSV)
    print(f"   Rows: {len(df)}")
    print(f"   Attack types:")
    print(df["attack_cat"].value_counts().to_string(header=False))

    # Convert UNSW features to our 25 features
    print("\n🔄 Converting UNSW-NB15 features to model input format...")
    X = np.vstack([convert_unsw_row(row) for _, row in df.iterrows()])
    print(f"   ✅ Converted: {X.shape}")

    # Run model on all candidates
    print("\n🔍 Running model on blind spot candidates...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs     = model(X_tensor)
        probs       = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = outputs.argmax(1).cpu().numpy()

    # Find blind spots = attacks model predicted as Benign (missed)
    # df["label"] = 1 means attack, prediction = 0 means model said benign
    actual_attacks = df["label"].values  # all should be 1 (attacks)
    missed_mask    = (predictions == 0)  # model said benign → missed

    blind_spots    = df[missed_mask].copy()
    blind_spots["predicted"]    = 0
    blind_spots["actual"]       = actual_attacks[missed_mask]
    blind_spots["confidence"]   = probs[missed_mask, 0]  # confidence it was benign
    blind_spots["X_features"]   = [X[i].tolist() for i in np.where(missed_mask)[0]]

    # Also save the converted features for missed attacks
    X_blind = X[missed_mask]

    print(f"\n📊 Results:")
    print(f"   Total candidates  : {len(df)}")
    print(f"   Correctly detected: {(~missed_mask).sum()} "
          f"({(~missed_mask).mean()*100:.1f}%)")
    print(f"   ⚠️  Blind spots     : {missed_mask.sum()} "
          f"({missed_mask.mean()*100:.1f}%) ← model missed these!")

    print(f"\n📊 Blind spots by attack type:")
    if len(blind_spots) > 0:
        print(blind_spots["attack_cat"].value_counts().to_string(header=False))
    else:
        print("   None! Model detected all attacks ✅")

    # Save blind spots
    if len(blind_spots) > 0:
        # Save full dataframe with attack_cat for mutation engine
        blind_spots_clean = df[missed_mask].copy()
        blind_spots_clean["confidence_missed"] = probs[missed_mask, 0]

        # Also save converted numeric features
        X_blind_df = pd.DataFrame(X_blind, columns=FEATURE_NAMES)
        X_blind_df["attack_cat"] = df[missed_mask]["attack_cat"].values
        X_blind_df["label"]      = 1  # all are attacks

        X_blind_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Blind spots saved to: {OUTPUT_CSV}")
        print(f"   {len(X_blind_df)} missed attacks ready for mutation engine")
    else:
        # If no blind spots found, save some hard attacks anyway
        # (lowest confidence attacks — hardest for model to detect)
        confidence_attack = probs[:, 1]  # confidence it's an attack
        hard_idx = np.argsort(confidence_attack)[:500]  # 500 hardest cases

        X_hard_df = pd.DataFrame(X[hard_idx], columns=FEATURE_NAMES)
        X_hard_df["attack_cat"] = df.iloc[hard_idx]["attack_cat"].values
        X_hard_df["label"]      = 1

        X_hard_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 No blind spots found — saving 500 hardest cases instead")
        print(f"   Saved to: {OUTPUT_CSV}")

    # Save log
    log = {
        "total_candidates":  int(len(df)),
        "correctly_detected": int((~missed_mask).sum()),
        "blind_spots_found": int(missed_mask.sum()),
        "detection_rate":    round(float((~missed_mask).mean() * 100), 2),
        "blind_spot_types":  blind_spots["attack_cat"].value_counts().to_dict()
                             if len(blind_spots) > 0 else {}
    }
    log_path = os.path.join(LOG_DIR, "blind_spot_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"📋 Log saved to: {log_path}")

    print(f"\n🎉 BLIND SPOT DETECTION DONE!")
    print(f"   Next step: Run mutation_engine.py")

if __name__ == "__main__":
    main()
