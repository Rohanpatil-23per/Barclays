"""
IMMUNEX - Layer 4: Mutation Engine
Takes blind spot attacks → creates 50 variations each
Input:  blind_spots.csv (missed attacks from blind_spot.py)
Output: mutated_attacks.csv (attack variants for retraining)
"""

import os
import json
import time
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = r"E:\immunex_p4\layer4_immunity"
LOG_DIR       = os.path.join(BASE_DIR, "logs")
BLIND_CSV     = os.path.join(BASE_DIR, "blind_spots.csv")
OUTPUT_CSV    = os.path.join(BASE_DIR, "mutated_attacks.csv")
LOG_PATH      = os.path.join(LOG_DIR,  "mutation_log.json")

os.makedirs(LOG_DIR, exist_ok=True)

# ─── 25 feature names ─────────────────────────────────────────────────────────
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

# ─── 5 Mutation Strategies ────────────────────────────────────────────────────
# Each strategy simulates a different way a hacker might change their attack
# to avoid detection

def strategy_timing_variation(row, variant):
    """
    Strategy 1 — Timing Variation
    Hackers speed up or slow down their attack timing
    Affects: flow_duration, flow_iat_mean, fwd_iat_mean, bwd_iat_mean
    10 variants: 5 fast (compress) + 5 slow (expand)
    """
    mutated = row.copy()
    timing_cols = [
        "flow_duration", "flow_iat_mean",
        "fwd_iat_mean",  "bwd_iat_mean"
    ]
    if variant < 5:
        # Fast attack — compress timing
        factor = np.random.uniform(0.1, 0.5)
    else:
        # Slow attack — expand timing (stealth)
        factor = np.random.uniform(1.5, 4.0)

    for col in timing_cols:
        if col in mutated:
            mutated[col] = mutated[col] * factor

    return mutated, "timing_variation"

def strategy_payload_obfuscation(row, variant):
    """
    Strategy 2 — Payload Obfuscation
    Hackers change packet sizes to look like normal traffic
    Affects: packet_length_mean, fwd_packet_length_mean,
             bwd_packet_length_mean, avg_packet_size
    10 variants with different size multipliers
    """
    mutated = row.copy()
    size_cols = [
        "packet_length_mean", "fwd_packet_length_mean",
        "bwd_packet_length_mean", "avg_packet_size", "packet_length_std"
    ]
    factors = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]
    factor  = factors[variant % len(factors)]

    for col in size_cols:
        if col in mutated:
            mutated[col] = mutated[col] * factor

    return mutated, "payload_obfuscation"

def strategy_evasion_noise(row, variant):
    """
    Strategy 3 — Evasion Noise
    Hackers add random fake traffic between attack packets
    to blend in with normal traffic
    Affects: all flow statistics with small random noise
    10 variants with different noise levels
    """
    mutated = row.copy()
    noise_levels = [0.01, 0.02, 0.05, 0.08, 0.1,
                    0.12, 0.15, 0.18, 0.2, 0.25]
    noise_level  = noise_levels[variant % len(noise_levels)]

    for col in FEATURE_NAMES:
        if col in mutated:
            noise = np.random.normal(0, abs(mutated[col]) * noise_level + 0.001)
            mutated[col] = mutated[col] + noise

    return mutated, "evasion_noise"

def strategy_lateral_movement(row, variant):
    """
    Strategy 4 — Lateral Movement
    Hacker moves from one system to another inside network
    Changes the ratio of forward/backward traffic
    Affects: total_fwd_packets, total_backward_packets,
             fwd_packets/s, bwd_packets/s, down/up_ratio
    10 variants with different movement patterns
    """
    mutated = row.copy()
    ratios  = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    ratio   = ratios[variant % len(ratios)]

    # Shift traffic direction
    if "total_fwd_packets" in mutated and "total_backward_packets" in mutated:
        total = abs(mutated["total_fwd_packets"]) + \
                abs(mutated["total_backward_packets"]) + 0.001
        mutated["total_fwd_packets"]      = total * ratio / (1 + ratio)
        mutated["total_backward_packets"] = total / (1 + ratio)

    if "fwd_packets/s" in mutated:
        mutated["fwd_packets/s"] = mutated["fwd_packets/s"] * ratio
    if "bwd_packets/s" in mutated:
        mutated["bwd_packets/s"] = mutated["bwd_packets/s"] / (ratio + 0.001)
    if "down/up_ratio" in mutated:
        mutated["down/up_ratio"] = mutated["down/up_ratio"] * ratio

    return mutated, "lateral_movement"

def strategy_tool_substitution(row, variant):
    """
    Strategy 5 — Tool Substitution
    Hacker switches to a different attack tool
    Different tools have different flag patterns and byte counts
    Affects: syn_flag_count, ack_flag_count, fin_flag_count,
             rst_flag_count, flow_bytes/s, flow_packets/s
    10 variants simulating different tools
    """
    mutated = row.copy()
    # Different tools produce different flag signatures
    tool_profiles = [
        {"syn": 1.0, "ack": 0.5, "fin": 0.2, "rst": 0.1},  # nmap
        {"syn": 0.5, "ack": 1.0, "fin": 0.5, "rst": 0.2},  # metasploit
        {"syn": 2.0, "ack": 0.2, "fin": 0.1, "rst": 0.5},  # hping3
        {"syn": 0.1, "ack": 2.0, "fin": 1.0, "rst": 0.3},  # scapy
        {"syn": 1.5, "ack": 1.5, "fin": 0.3, "rst": 0.8},  # zmap
        {"syn": 0.3, "ack": 0.3, "fin": 2.0, "rst": 1.5},  # masscan
        {"syn": 3.0, "ack": 0.1, "fin": 0.1, "rst": 0.1},  # slowloris
        {"syn": 0.2, "ack": 3.0, "fin": 0.2, "rst": 0.2},  # goldeneye
        {"syn": 1.0, "ack": 1.0, "fin": 1.0, "rst": 1.0},  # mixed
        {"syn": 0.5, "ack": 0.5, "fin": 0.5, "rst": 3.0},  # rst flood
    ]
    profile = tool_profiles[variant % len(tool_profiles)]

    if "syn_flag_count" in mutated:
        mutated["syn_flag_count"] = mutated["syn_flag_count"] * profile["syn"]
    if "ack_flag_count" in mutated:
        mutated["ack_flag_count"] = mutated["ack_flag_count"] * profile["ack"]
    if "fin_flag_count" in mutated:
        mutated["fin_flag_count"] = mutated["fin_flag_count"] * profile["fin"]
    if "rst_flag_count" in mutated:
        mutated["rst_flag_count"] = mutated["rst_flag_count"] * profile["rst"]

    # Different tools also have different byte rates
    byte_factor = np.random.uniform(0.3, 2.5)
    if "flow_bytes/s" in mutated:
        mutated["flow_bytes/s"] = mutated["flow_bytes/s"] * byte_factor

    return mutated, "tool_substitution"

# ─── Apply all 5 strategies to one attack ─────────────────────────────────────
STRATEGIES = [
    strategy_timing_variation,
    strategy_payload_obfuscation,
    strategy_evasion_noise,
    strategy_lateral_movement,
    strategy_tool_substitution,
]

def mutate_one_attack(row_dict, attack_cat):
    """
    Takes one attack record
    Applies all 5 strategies × 10 variants each = 50 mutations
    Returns list of 50 mutated rows
    """
    mutations = []
    for strategy_fn in STRATEGIES:
        for variant in range(10):
            mutated, strategy_name = strategy_fn(row_dict.copy(), variant)
            mutated["mutation_label"]    = 1          # still an attack
            mutated["attack_cat"]        = attack_cat
            mutated["mutation_strategy"] = strategy_name
            mutated["mutation_variant"]  = variant
            mutations.append(mutated)
    return mutations  # 50 mutations per attack

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4: MUTATION ENGINE")
    print("="*60)

    # Load blind spots
    print("\n📂 Loading blind spots...")
    df = pd.read_csv(BLIND_CSV)
    print(f"   Total blind spots: {len(df)}")
    print(f"   Attack types:")
    print(df["attack_cat"].value_counts().to_string(header=False))

    # Sample attacks to mutate
    # We sample up to 200 per attack type to keep it manageable
    sampled_frames = []
    for cat in df["attack_cat"].unique():
        subset = df[df["attack_cat"] == cat]
        n      = min(len(subset), 200)
        sampled_frames.append(subset.sample(n, random_state=42))
    sampled = pd.concat(sampled_frames).reset_index(drop=True)

    print(f"\n✅ Sampled {len(sampled)} attacks for mutation")
    print(f"   Expected mutations: {len(sampled) * 50:,} "
          f"(50 per attack = 5 strategies × 10 variants)")

    # Run mutation engine
    print(f"\n🔄 Running mutation engine...")
    all_mutations = []
    strategy_counts = {}
    t0 = time.time()

    for i, (_, row) in enumerate(sampled.iterrows()):
        # Build feature dict
        row_dict   = {f: float(row[f]) if f in row.index else 0.0
                      for f in FEATURE_NAMES}
        attack_cat = row.get("attack_cat", "Unknown")

        # Generate 50 mutations
        mutations  = mutate_one_attack(row_dict, attack_cat)
        all_mutations.extend(mutations)

        # Count per strategy
        for m in mutations:
            s = m["mutation_strategy"]
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        # Progress update
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  🔄 {i+1}/{len(sampled)} attacks | "
                  f"Mutations: {len(all_mutations):,} | "
                  f"Time: {elapsed:.0f}s")

    elapsed = time.time() - t0

    # Build final dataframe
    print(f"\n💾 Building output dataframe...")
    mutations_df = pd.DataFrame(all_mutations)

    # Keep only feature columns + metadata
    keep_cols = FEATURE_NAMES + [
        "mutation_label", "attack_cat",
        "mutation_strategy", "mutation_variant"
    ]
    mutations_df = mutations_df[[c for c in keep_cols
                                  if c in mutations_df.columns]]

    # Save
    mutations_df.to_csv(OUTPUT_CSV, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  MUTATION ENGINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ Total mutations generated: {len(mutations_df):,}")
    print(f"📊 Breakdown by strategy:")
    for strategy, count in strategy_counts.items():
        print(f"   {strategy}: {count:,} variants")
    print(f"📊 Breakdown by attack type:")
    print(mutations_df["attack_cat"].value_counts().to_string(header=False))
    print(f"\n⏱️  Time taken: {elapsed:.1f}s")
    print(f"💾 Saved to: {OUTPUT_CSV}")

    # Save log
    log = {
        "total_blind_spots":   int(len(df)),
        "sampled_for_mutation": int(len(sampled)),
        "total_mutations":     int(len(mutations_df)),
        "strategy_counts":     strategy_counts,
        "attack_type_counts":  mutations_df["attack_cat"]
                               .value_counts().to_dict(),
        "time_seconds":        round(elapsed, 1)
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"📋 Log saved to: {LOG_PATH}")

    print(f"\n🎉 MUTATION ENGINE DONE!")
    print(f"   Next step: Run ewc.py")

if __name__ == "__main__":
    main()
