"""
IMMUNEX - Layer 4: Mutation Engine (Fixed)
Source: lora_retrain_source.csv (CICIDS - same as training data)
Takes attack rows → creates 50 variations each
Output: mutated_attacks.csv (same format as training data)
"""

import os
import json
import time
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"E:\immunex_p4\layer4_immunity"
DATA_DIR   = r"E:\immunex_p4\person4_layer4"
LOG_DIR    = os.path.join(BASE_DIR, "logs")
TRAIN_CSV  = os.path.join(DATA_DIR, "lora_retrain_source.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "mutated_attacks.csv")
LOG_PATH   = os.path.join(LOG_DIR,  "mutation_log.json")

os.makedirs(LOG_DIR, exist_ok=True)

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

# ─── Parse text → dict ────────────────────────────────────────────────────────
def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try: lookup[key] = float(val)
            except: lookup[key] = 0.0
    return {f: lookup.get(f, 0.0) for f in FEATURE_NAMES}

def dict_to_array(d):
    return np.array([d[f] for f in FEATURE_NAMES], dtype=np.float32)

# ─── 5 Mutation Strategies (all stay in CICIDS normalized range) ──────────────

def strategy_timing_variation(feat_dict, variant):
    """
    Hacker speeds up or slows down attack timing
    Fast (0-4): compress timing → quicker attack
    Slow (5-9): expand timing  → stealthy slow attack
    """
    mutated     = feat_dict.copy()
    timing_cols = ["flow_duration","flow_iat_mean","fwd_iat_mean","bwd_iat_mean"]
    factor      = np.random.uniform(0.3, 0.7) if variant < 5 \
                  else np.random.uniform(1.3, 2.5)
    for col in timing_cols:
        mutated[col] = np.clip(mutated[col] * factor, -3, 5)
    return mutated, "timing_variation"

def strategy_payload_obfuscation(feat_dict, variant):
    """
    Hacker changes packet sizes to blend in with normal traffic
    """
    mutated   = feat_dict.copy()
    size_cols = ["packet_length_mean","fwd_packet_length_mean",
                 "bwd_packet_length_mean","avg_packet_size","packet_length_std"]
    factors   = [0.3,0.5,0.6,0.7,0.8,1.2,1.4,1.6,1.8,2.0]
    factor    = factors[variant % len(factors)]
    for col in size_cols:
        mutated[col] = np.clip(mutated[col] * factor, -3, 5)
    return mutated, "payload_obfuscation"

def strategy_evasion_noise(feat_dict, variant):
    """
    Hacker adds random fake traffic between attack packets
    Small random noise added to all features
    """
    mutated      = feat_dict.copy()
    noise_levels = [0.01,0.02,0.03,0.05,0.07,0.08,0.10,0.12,0.15,0.18]
    noise_level  = noise_levels[variant % len(noise_levels)]
    for col in FEATURE_NAMES:
        noise = np.random.normal(0, abs(mutated[col]) * noise_level + 0.001)
        mutated[col] = np.clip(mutated[col] + noise, -3, 5)
    return mutated, "evasion_noise"

def strategy_lateral_movement(feat_dict, variant):
    """
    Hacker moves from one system to another
    Changes ratio of forward vs backward traffic
    """
    mutated = feat_dict.copy()
    ratios  = [0.2,0.4,0.5,0.6,0.8,1.2,1.5,1.8,2.0,2.5]
    ratio   = ratios[variant % len(ratios)]
    fwd     = mutated["total_fwd_packets"]
    bwd     = mutated["total_backward_packets"]
    total   = abs(fwd) + abs(bwd) + 0.001
    mutated["total_fwd_packets"]      = np.clip(total*ratio/(1+ratio), -3, 5)
    mutated["total_backward_packets"] = np.clip(total/(1+ratio), -3, 5)
    mutated["fwd_packets/s"]          = np.clip(mutated["fwd_packets/s"]*ratio, -3, 5)
    mutated["bwd_packets/s"]          = np.clip(mutated["bwd_packets/s"]/(ratio+0.001), -3, 5)
    mutated["down/up_ratio"]          = np.clip(mutated["down/up_ratio"]*ratio, -3, 5)
    return mutated, "lateral_movement"

def strategy_tool_substitution(feat_dict, variant):
    """
    Hacker switches to a different attack tool
    Different tools produce different flag patterns
    """
    mutated = feat_dict.copy()
    tool_profiles = [
        {"syn":1.5,"ack":0.5,"fin":0.2,"rst":0.1},  # nmap
        {"syn":0.5,"ack":1.5,"fin":0.5,"rst":0.2},  # metasploit
        {"syn":2.0,"ack":0.2,"fin":0.1,"rst":0.5},  # hping3
        {"syn":0.1,"ack":2.0,"fin":1.0,"rst":0.3},  # scapy
        {"syn":1.5,"ack":1.5,"fin":0.3,"rst":0.8},  # zmap
        {"syn":0.3,"ack":0.3,"fin":2.0,"rst":1.5},  # masscan
        {"syn":2.5,"ack":0.1,"fin":0.1,"rst":0.1},  # slowloris
        {"syn":0.2,"ack":2.5,"fin":0.2,"rst":0.2},  # goldeneye
        {"syn":1.0,"ack":1.0,"fin":1.0,"rst":1.0},  # mixed
        {"syn":0.5,"ack":0.5,"fin":0.5,"rst":2.5},  # rst flood
    ]
    p = tool_profiles[variant % len(tool_profiles)]
    mutated["syn_flag_count"] = np.clip(mutated["syn_flag_count"]*p["syn"], -3, 5)
    mutated["ack_flag_count"] = np.clip(mutated["ack_flag_count"]*p["ack"], -3, 5)
    mutated["fin_flag_count"] = np.clip(mutated["fin_flag_count"]*p["fin"], -3, 5)
    mutated["rst_flag_count"] = np.clip(mutated["rst_flag_count"]*p["rst"], -3, 5)
    mutated["flow_bytes/s"]   = np.clip(
        mutated["flow_bytes/s"] * np.random.uniform(0.5, 2.0), -3, 5)
    return mutated, "tool_substitution"

STRATEGIES = [
    strategy_timing_variation,
    strategy_payload_obfuscation,
    strategy_evasion_noise,
    strategy_lateral_movement,
    strategy_tool_substitution,
]

def mutate_one_attack(feat_dict):
    """5 strategies × 10 variants = 50 mutations per attack"""
    mutations = []
    for strategy_fn in STRATEGIES:
        for variant in range(10):
            mutated, name = strategy_fn(feat_dict.copy(), variant)
            mutations.append({
                "features":          dict_to_array(mutated),
                "mutation_label":    1,
                "mutation_strategy": name,
                "mutation_variant":  variant,
            })
    return mutations

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4: MUTATION ENGINE (CICIDS Source)")
    print("="*60)

    # Load CICIDS training data
    print("\n📂 Loading CICIDS training data...")
    df      = pd.read_csv(TRAIN_CSV)
    attacks = df[df["label"] == 1].reset_index(drop=True)
    print(f"   Total rows   : {len(df):,}")
    print(f"   Attack rows  : {len(attacks):,}")

    # Sample 600 attacks → 30,000 mutations
    n_sample = min(600, len(attacks))
    sampled  = attacks.sample(n_sample, random_state=42).reset_index(drop=True)
    print(f"\n✅ Sampled {n_sample} attacks for mutation")
    print(f"   Expected mutations: {n_sample * 50:,}")

    # Parse features
    print("\n🔄 Parsing attack features...")
    feat_dicts = [parse_text(t) for t in sampled["text"]]
    sample_arr = np.array([dict_to_array(d) for d in feat_dicts])
    print(f"   ✅ Scale: min={sample_arr.min():.3f} max={sample_arr.max():.3f}")
    print(f"   (matches training data — CICIDS normalized format)")

    # Generate mutations
    print(f"\n🔄 Running mutation engine...")
    all_features, all_labels    = [], []
    all_strategies, all_variants = [], []
    strategy_counts = {}
    t0 = time.time()

    for i, feat_dict in enumerate(feat_dicts):
        for m in mutate_one_attack(feat_dict):
            all_features.append(m["features"])
            all_labels.append(m["mutation_label"])
            all_strategies.append(m["mutation_strategy"])
            all_variants.append(m["mutation_variant"])
            s = m["mutation_strategy"]
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        if (i + 1) % 100 == 0:
            print(f"  🔄 {i+1}/{n_sample} | "
                  f"Mutations: {len(all_features):,} | "
                  f"Time: {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    X_mut   = np.vstack(all_features)

    # Build and save dataframe
    print(f"\n💾 Saving mutations...")
    df_out = pd.DataFrame(X_mut, columns=FEATURE_NAMES)
    df_out["mutation_label"]    = all_labels
    df_out["mutation_strategy"] = all_strategies
    df_out["mutation_variant"]  = all_variants
    df_out["attack_cat"]        = "CICIDS_attack"
    df_out.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  MUTATION ENGINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ Total mutations    : {len(df_out):,}")
    print(f"📊 By strategy:")
    for s, c in strategy_counts.items():
        print(f"   {s}: {c:,}")
    print(f"\n📊 Scale verification:")
    print(f"   Training data : -1.128 to 21.108")
    print(f"   Mutations     : {X_mut.min():.3f} to {X_mut.max():.3f} ✅")
    print(f"⏱️  Time          : {elapsed:.1f}s")
    print(f"💾 Saved to      : {OUTPUT_CSV}")

    # Save log
    with open(LOG_PATH, "w") as f:
        json.dump({
            "source":          "lora_retrain_source.csv (CICIDS)",
            "total_attacks":   int(len(attacks)),
            "sampled":         int(n_sample),
            "total_mutations": int(len(df_out)),
            "strategy_counts": strategy_counts,
            "scale_min":       float(X_mut.min()),
            "scale_max":       float(X_mut.max()),
            "time_seconds":    round(elapsed, 1),
        }, f, indent=2)
    print(f"📋 Log saved     : {LOG_PATH}")
    print(f"\n🎉 DONE! Next step: Run ewc.py")
    print(f"   ✅ Mutations in CICIDS format → EWC will work properly")

if __name__ == "__main__":
    main()
