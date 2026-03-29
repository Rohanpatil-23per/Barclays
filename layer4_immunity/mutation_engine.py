"""
IMMUNEX - Layer 4: RL Mutation Engine with Attack Combination
=============================================================
What this file does:
  Generates 30,000 mutated attack variants for EWC retraining.
  Uses THREE generation modes so the model learns ALL evasion styles:

  Mode 1 — RL Single-Attack Mutations (15,000 samples)
    The RL agent (AttackerPolicy) plays attacker vs your detector.
    Agent sees one attack sample, learns which 2-5 features to perturb
    together to make the detector say "Benign" (bypass).
    Reward = +1 bypass / -0.1 caught.
    Over 2000 episodes the agent discovers YOUR model's specific blind spots.

  Mode 2 — Attack Interpolation (7,500 samples)
    Blends two different attack samples together:
      mutated = alpha * Attack_A  +  (1 - alpha) * Attack_B
      alpha ~ Uniform(0.3, 0.7)
    Simulates an attacker switching between two techniques mid-campaign.
    The RL agent then further perturbs each blend.
    These sit in feature-space regions the detector has never seen.

  Mode 3 — Feature Splicing (7,500 samples)
    Takes 2-3 attack samples and splices their feature groups:
      Timing features  (flow_duration, IAT)      → from Attack A
      Payload features (packet lengths, bytes)   → from Attack B
      Flag features    (SYN, ACK, FIN, RST, PSH) → from Attack C
    Simulates an attacker using one tool's timing with another tool's
    packet sizes — e.g. nmap timing + metasploit payload obfuscation.

Why this matters:
  Your detector was trained on pure, isolated attack signatures.
  Hybrid attacks (blend + splice) sit in unexplored feature-space gaps.
  Training on all 3 modes closes those gaps.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = BASE_DIR
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lora_model.pt")
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
N_FEAT = len(FEATURE_NAMES)  # 25

# Feature group indices for splicing (Mode 3)
# These group semantically related features together
TIMING_IDX  = [0, 7, 8, 9, 21, 22]          # flow_duration, IAT features, active/idle
PAYLOAD_IDX = [5, 6, 15, 16, 24, 3, 4]      # packet lengths, bytes/s, packet/s
FLAG_IDX    = [10, 11, 12, 13, 14]           # SYN, ACK, FIN, RST, PSH flags
FLOW_IDX    = [1, 2, 17, 18, 19, 20, 23]    # packet counts, win bytes, ratio


# ─── Detector model (same architecture as lora_retrain.py) ───────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base   = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

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


# ─── RL Attacker Agent ────────────────────────────────────────────────────────
class AttackerPolicy(nn.Module):
    """
    RL agent that learns which COMBINATION of features to perturb.

    Input  : attack sample (25 features) = state
    Output :
      feature_probs → Bernoulli probabilities over 25 features
                      agent LEARNS which feature combos fool detector
      delta_mean    → how much to shift each selected feature
      log_std       → learnable uncertainty
      value         → critic baseline for variance reduction

    Key point: agent picks 2-5 features simultaneously (not one at a time).
    The combination is what's dynamic — e.g. it discovers that changing
    flow_bytes/s + syn_flag_count + idle_mean TOGETHER fools your model,
    whereas changing any one of them alone does not.
    """
    def __init__(self, state_dim=25, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.feature_logits = nn.Linear(hidden, state_dim)
        self.delta_mean     = nn.Linear(hidden, state_dim)
        self.log_std        = nn.Parameter(torch.full((state_dim,), -1.0))
        self.value_head     = nn.Linear(hidden, 1)

    def forward(self, state):
        h             = self.shared(state)
        feature_probs = torch.sigmoid(self.feature_logits(h))
        delta_mean    = self.delta_mean(h)
        value         = self.value_head(h).squeeze(-1)
        return feature_probs, delta_mean, self.log_std.exp(), value

    def select_action(self, state):
        """
        Samples which features to perturb AND by how much.
        Enforces 2-5 features per mutation.
        """
        feature_probs, delta_mean, delta_std, value = self.forward(state)
        B = state.shape[0]

        feature_mask = Bernoulli(feature_probs).sample()

        # Enforce minimum 2 features
        n_sel = feature_mask.sum(dim=-1)
        for i in range(B):
            if n_sel[i] < 2:
                top2 = feature_probs[i].topk(2).indices
                feature_mask[i] = torch.zeros(N_FEAT, device=state.device)
                feature_mask[i][top2] = 1.0

        # Enforce maximum 5 features
        for i in range(B):
            if feature_mask[i].sum() > 5:
                top5 = feature_probs[i].topk(5).indices
                feature_mask[i] = torch.zeros(N_FEAT, device=state.device)
                feature_mask[i][top5] = 1.0

        dist     = Normal(delta_mean, delta_std.unsqueeze(0).expand_as(delta_mean))
        delta    = dist.rsample()
        action   = delta * feature_mask

        log_p_mask  = (feature_probs * feature_mask +
                       (1 - feature_probs) * (1 - feature_mask)).clamp(1e-8).log().sum(-1)
        log_p_delta = dist.log_prob(delta).sum(-1)
        log_prob    = log_p_mask + log_p_delta

        return action, log_prob, value, feature_mask


# ─── Mode 2: Attack Interpolation ────────────────────────────────────────────
def generate_interpolations(
        X_attacks: np.ndarray,
        agent:     "AttackerPolicy",
        detector:  nn.Module,
        device:    torch.device,
        target:    int   = 7500,
        perturb_scale: float = 0.3,
) -> tuple:
    """
    Blends pairs of attack samples then applies RL perturbation.

    For each pair (A, B):
      alpha  ~ Uniform(0.3, 0.7)   — random blend ratio
      blend  = alpha * A + (1-alpha) * B
      mutated = blend + RL_agent_perturbation(blend)

    Why 0.3-0.7 range: ensures both attacks contribute meaningfully.
    Pure A (alpha=1.0) or pure B (alpha=0.0) would just be single-attack.

    Returns (mutations array, bypass flags array).
    """
    print(f"\n  [Mode 2] Generating {target:,} interpolation hybrids...")
    all_muts   = []
    all_bypass = []
    X_t        = torch.FloatTensor(X_attacks).to(device)
    agent.eval()
    detector.eval()
    generated  = 0

    while generated < target:
        bsz = min(256, target - generated + 32)

        # Pick two independent random attack batches
        idx_a = np.random.choice(len(X_attacks), bsz, replace=True)
        idx_b = np.random.choice(len(X_attacks), bsz, replace=True)
        A = X_t[idx_a]
        B = X_t[idx_b]

        # Random blend ratios — different per sample
        alpha = torch.FloatTensor(bsz, 1).uniform_(0.3, 0.7).to(device)
        blend = alpha * A + (1 - alpha) * B   # (bsz, 25)

        # RL agent further perturbs the blend
        with torch.no_grad():
            actions, _, _, _ = agent.select_action(blend)
            mutated = (blend + actions * perturb_scale).clamp(-5.0, 20.0)
            preds   = detector(mutated).argmax(dim=-1)

        all_muts.append(mutated.cpu().numpy())
        all_bypass.extend((preds == 0).cpu().numpy().tolist())
        generated += bsz

    result = np.concatenate(all_muts, axis=0)[:target]
    bypass = np.array(all_bypass[:target])
    n_bypass = bypass.sum()
    print(f"  [Mode 2] Done. {len(result):,} interpolations | "
          f"Bypasses: {n_bypass:,} ({bypass.mean():.1%})")
    return result, bypass


# ─── Mode 3: Feature Splicing ─────────────────────────────────────────────────
def generate_spliced(
        X_attacks: np.ndarray,
        agent:     "AttackerPolicy",
        detector:  nn.Module,
        device:    torch.device,
        target:    int   = 7500,
        perturb_scale: float = 0.3,
) -> tuple:
    """
    Splices feature groups from 2-3 different attack samples.

    Feature groups (defined at top of file):
      TIMING_IDX  → flow_duration, IAT features, active/idle mean
      PAYLOAD_IDX → packet lengths, bytes/s
      FLAG_IDX    → SYN, ACK, FIN, RST, PSH flags
      FLOW_IDX    → packet counts, window bytes, ratio

    Example chimera:
      Timing  from Attack A  (e.g. fast SSH scan timing)
      Payload from Attack B  (e.g. FTP's packet sizes)
      Flags   from Attack C  (e.g. Bot's flag pattern)

    This is realistic: an attacker using nmap's timing profile but
    spoofing FTP packet sizes to confuse signature-based detectors.

    Two splice patterns used:
      2-source: groups split between 2 attacks (50/50)
      3-source: each of 4 groups comes from a different attack

    Returns (mutations array, bypass flags array).
    """
    print(f"\n  [Mode 3] Generating {target:,} spliced chimeras...")
    groups    = [TIMING_IDX, PAYLOAD_IDX, FLAG_IDX, FLOW_IDX]
    all_muts  = []
    all_bypass = []
    X_t        = torch.FloatTensor(X_attacks).to(device)
    agent.eval()
    detector.eval()
    generated  = 0

    while generated < target:
        bsz = min(256, target - generated + 32)

        # Build chimera array on CPU first
        chimeras = np.zeros((bsz, N_FEAT), dtype=np.float32)

        # Randomly choose 2-source or 3-source splice per sample
        use_3src = np.random.rand(bsz) > 0.5

        for i in range(bsz):
            if use_3src[i]:
                # 3-source: pick 3 different attacks, assign groups round-robin
                srcs = [X_attacks[np.random.randint(len(X_attacks))] for _ in range(3)]
                chimeras[i] = srcs[0].copy()               # start with attack A
                for feat_idx in PAYLOAD_IDX:
                    chimeras[i, feat_idx] = srcs[1][feat_idx]  # payload from B
                for feat_idx in FLAG_IDX:
                    chimeras[i, feat_idx] = srcs[2][feat_idx]  # flags from C
            else:
                # 2-source: two attacks, alternate groups between them
                src_a = X_attacks[np.random.randint(len(X_attacks))]
                src_b = X_attacks[np.random.randint(len(X_attacks))]
                chimeras[i] = src_a.copy()                  # start with A
                # Overwrite half the groups with B
                for feat_idx in PAYLOAD_IDX + FLAG_IDX:
                    chimeras[i, feat_idx] = src_b[feat_idx]

        chimera_t = torch.FloatTensor(chimeras).to(device)

        # RL agent further perturbs the chimera
        with torch.no_grad():
            actions, _, _, _ = agent.select_action(chimera_t)
            mutated = (chimera_t + actions * perturb_scale).clamp(-5.0, 20.0)
            preds   = detector(mutated).argmax(dim=-1)

        all_muts.append(mutated.cpu().numpy())
        all_bypass.extend((preds == 0).cpu().numpy().tolist())
        generated += bsz

    result = np.concatenate(all_muts, axis=0)[:target]
    bypass = np.array(all_bypass[:target])
    n_bypass = bypass.sum()
    print(f"  [Mode 3] Done. {len(result):,} spliced chimeras | "
          f"Bypasses: {n_bypass:,} ({bypass.mean():.1%})")
    return result, bypass


# ─── Mode 1: RL Single-Attack Mutations ──────────────────────────────────────
def train_rl_agent(
        X_attacks:     np.ndarray,
        detector:      nn.Module,
        device:        torch.device,
        n_episodes:    int   = 2000,
        batch_size:    int   = 64,
        lr:            float = 3e-4,
        perturb_scale: float = 0.4,
) -> tuple:
    """
    Trains the RL attacker via REINFORCE with baseline.

    Each episode:
      1. Sample batch of real attack samples as starting states
      2. Agent proposes which 2-5 features to perturb and by how much
      3. Apply perturbations → mutated samples
      4. Query detector: Benign prediction = bypass (reward +1), Attack = caught (-0.1)
      5. Policy gradient: increase probability of actions that caused bypasses

    Returns trained agent + collected bypass samples.
    """
    agent     = AttackerPolicy(state_dim=N_FEAT).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    all_bypasses     = []
    all_masks_used   = []
    bypass_rate_hist = []
    X_t              = torch.FloatTensor(X_attacks).to(device)

    print(f"\n  Training RL attacker agent ({n_episodes} episodes)...")
    print(f"  Reward: +1 if detector says Benign (bypass), -0.1 if caught")
    print(f"  Agent learns which FEATURE COMBINATIONS fool YOUR detector\n")

    for ep in range(n_episodes):
        idx    = np.random.choice(len(X_attacks), batch_size, replace=True)
        states = X_t[idx]

        agent.train()
        actions, log_probs, values, feature_masks = agent.select_action(states)

        mutated = (states + actions * perturb_scale).clamp(-5.0, 20.0)

        detector.eval()
        with torch.no_grad():
            logits = detector(mutated)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

        bypass_conf = probs[:, 0]
        rewards = torch.where(
            preds == 0,
            0.5 + 0.5 * bypass_conf,
            -0.1 * (1.0 - bypass_conf)
        )

        bypass_idx = (preds == 0).nonzero(as_tuple=True)[0]
        if len(bypass_idx) > 0:
            all_bypasses.append(mutated[bypass_idx].detach().cpu().numpy())
            all_masks_used.append(feature_masks[bypass_idx].detach().cpu().numpy())

        advantages = (rewards - values.detach())
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = (rewards - values).pow(2).mean()
        entropy     = -(log_probs.detach() * log_probs).mean() * 0.01
        loss        = actor_loss + 0.5 * critic_loss - entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

        bypass_rate_hist.append(len(bypass_idx) / batch_size)

        if (ep + 1) % 400 == 0:
            recent_rate = np.mean(bypass_rate_hist[-200:])
            n_collected = sum(len(b) for b in all_bypasses)
            print(f"  Episode {ep+1:4d}/{n_episodes} | "
                  f"Bypass rate: {recent_rate:.1%} | "
                  f"Collected bypasses: {n_collected:,}")

    return agent, all_bypasses, all_masks_used


def generate_rl_single(
        agent:         AttackerPolicy,
        X_attacks:     np.ndarray,
        detector:      nn.Module,
        device:        torch.device,
        target:        int   = 15000,
        perturb_scale: float = 0.4,
) -> tuple:
    """Generates single-attack RL mutations using the trained agent."""
    print(f"\n  [Mode 1] Generating {target:,} single-attack RL mutations...")
    all_muts   = []
    all_bypass = []
    X_t        = torch.FloatTensor(X_attacks).to(device)
    agent.eval()
    detector.eval()
    generated  = 0

    while generated < target:
        bsz    = min(512, target - generated + 64)
        idx    = np.random.choice(len(X_attacks), bsz, replace=True)
        states = X_t[idx]

        with torch.no_grad():
            actions, _, _, _ = agent.select_action(states)
            mutated = (states + actions * perturb_scale).clamp(-5.0, 20.0)
            preds   = detector(mutated).argmax(dim=-1)

        all_muts.append(mutated.cpu().numpy())
        all_bypass.extend((preds == 0).cpu().numpy().tolist())
        generated += bsz

        if generated % 5000 < bsz:
            n_bp = sum(all_bypass)
            print(f"  [Mode 1] {generated:,}/{target:,} | "
                  f"Bypasses: {n_bp:,} ({n_bp/max(1,len(all_bypass)):.1%})")

    result = np.concatenate(all_muts, axis=0)[:target]
    bypass = np.array(all_bypass[:target])
    print(f"  [Mode 1] Done. {len(result):,} single-attack mutations | "
          f"Bypasses: {bypass.sum():,} ({bypass.mean():.1%})")
    return result, bypass


# ─── Feature importance ───────────────────────────────────────────────────────
def get_feature_importance(agent, X_attacks, device):
    agent.eval()
    X_t = torch.FloatTensor(X_attacks[:1000]).to(device)
    with torch.no_grad():
        probs, _, _, _ = agent.forward(X_t)
    mean_probs = probs.mean(dim=0).cpu().numpy()
    return sorted(zip(FEATURE_NAMES, mean_probs), key=lambda x: x[1], reverse=True)


# ─── Data helpers ─────────────────────────────────────────────────────────────
def parse_text(text):
    lookup = {}
    for pair in text.strip().split():
        if ":" in pair:
            key, val = pair.split(":", 1)
            try:    lookup[key] = float(val)
            except: lookup[key] = 0.0
    return {f: lookup.get(f, 0.0) for f in FEATURE_NAMES}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  IMMUNEX - LAYER 4: RL MUTATION ENGINE (3 MODES)")
    print("  Mode 1: RL single-attack    (15,000 samples)")
    print("  Mode 2: Attack interpolation (7,500 samples)")
    print("  Mode 3: Feature splicing     (7,500 samples)")
    print("=" * 60)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load detector ─────────────────────────────────────────────────────────
    print("\n  Loading trained detector...")
    ckpt     = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    rank     = ckpt.get("lora_rank", 8)
    detector = IMMUNEXLayer4(input_dim=25, rank=rank).to(device)
    detector.load_state_dict(ckpt["model_state"])
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False
    print(f"  Detector loaded | Rank={rank} | Accuracy: {ckpt['accuracy']:.2f}%")

    # ── Load CICIDS attack samples ─────────────────────────────────────────────
    print("\n  Loading CICIDS attack samples (label=1)...")
    df        = pd.read_csv(TRAIN_CSV)
    atk_df    = df[df["label"] == 1].sample(
        min(2000, int((df["label"] == 1).sum())), random_state=42
    )
    X_attacks = np.array(
        [list(parse_text(t).values()) for t in atk_df["text"]],
        dtype=np.float32
    )
    print(f"  Attack samples : {len(X_attacks):,}")
    print(f"  Scale          : {X_attacks.min():.3f} to {X_attacks.max():.3f}")

    t0 = time.time()

    # ── Train RL agent (used by all 3 modes) ──────────────────────────────────
    agent, bypass_pool, mask_pool = train_rl_agent(
        X_attacks     = X_attacks,
        detector      = detector,
        device        = device,
        n_episodes    = 2000,
        batch_size    = 64,
        perturb_scale = 0.4,
    )

    # ── Mode 1: Single-attack RL mutations (15,000) ───────────────────────────
    print("\n" + "─" * 50)
    print("  MODE 1 — Single-Attack RL Mutations")
    print("─" * 50)
    m1_arr, m1_bypass = generate_rl_single(
        agent=agent, X_attacks=X_attacks, detector=detector,
        device=device, target=15000, perturb_scale=0.4
    )

    # ── Mode 2: Attack interpolation (7,500) ─────────────────────────────────
    print("\n" + "─" * 50)
    print("  MODE 2 — Attack Interpolation (blend two attacks)")
    print("─" * 50)
    m2_arr, m2_bypass = generate_interpolations(
        X_attacks=X_attacks, agent=agent, detector=detector,
        device=device, target=7500, perturb_scale=0.3
    )

    # ── Mode 3: Feature splicing (7,500) ─────────────────────────────────────
    print("\n" + "─" * 50)
    print("  MODE 3 — Feature Splicing (chimera from 2-3 attacks)")
    print("─" * 50)
    m3_arr, m3_bypass = generate_spliced(
        X_attacks=X_attacks, agent=agent, detector=detector,
        device=device, target=7500, perturb_scale=0.3
    )

    elapsed = time.time() - t0

    # ── Combine all 3 modes ───────────────────────────────────────────────────
    all_mutations = np.vstack([m1_arr, m2_arr, m3_arr])    # (30000, 25)
    all_bypass    = np.concatenate([m1_bypass, m2_bypass, m3_bypass])
    all_strategy  = (["rl_single"]       * len(m1_arr) +
                     ["interpolation"]   * len(m2_arr) +
                     ["feature_splice"]  * len(m3_arr))

    # ── Feature importance ────────────────────────────────────────────────────
    importance_ranked = get_feature_importance(agent, X_attacks, device)
    print("\n  Top 5 features the agent learned to exploit:")
    for feat, prob in importance_ranked[:5]:
        bar = "█" * int(prob * 20)
        print(f"    {feat:<35} {prob:.3f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n  Saving 30,000 mutations...")
    df_out = pd.DataFrame(all_mutations, columns=FEATURE_NAMES)
    df_out["mutation_label"]    = 1
    df_out["mutation_strategy"] = all_strategy
    df_out["bypass_detector"]   = all_bypass.astype(int)
    df_out["attack_cat"]        = "CICIDS_attack"
    df_out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_bypass = int(all_bypass.sum())
    print("\n" + "=" * 60)
    print("  RL MUTATION ENGINE COMPLETE")
    print("=" * 60)
    print(f"  Total mutations   : {len(df_out):,}")
    print(f"  Mode 1 (RL)       : {len(m1_arr):,}  | "
          f"Bypasses: {m1_bypass.sum():,} ({m1_bypass.mean():.1%})")
    print(f"  Mode 2 (Interp.)  : {len(m2_arr):,}  | "
          f"Bypasses: {m2_bypass.sum():,} ({m2_bypass.mean():.1%})")
    print(f"  Mode 3 (Splice)   : {len(m3_arr):,}  | "
          f"Bypasses: {m3_bypass.sum():,} ({m3_bypass.mean():.1%})")
    print(f"  Total bypasses    : {total_bypass:,} ({all_bypass.mean():.1%})")
    print(f"  Scale             : {all_mutations.min():.3f} to "
          f"{all_mutations.max():.3f} ✅")
    print(f"  Time              : {elapsed:.1f}s")
    print(f"  Saved to          : {OUTPUT_CSV}")

    log = {
        "approach":   "RL agent + interpolation + feature splicing",
        "source":     "CICIDS lora_retrain_source.csv (label=1)",
        "modes": {
            "rl_single":     {"count": len(m1_arr),
                              "bypasses": int(m1_bypass.sum()),
                              "bypass_rate": round(float(m1_bypass.mean()), 4)},
            "interpolation": {"count": len(m2_arr),
                              "bypasses": int(m2_bypass.sum()),
                              "bypass_rate": round(float(m2_bypass.mean()), 4)},
            "feature_splice":{"count": len(m3_arr),
                              "bypasses": int(m3_bypass.sum()),
                              "bypass_rate": round(float(m3_bypass.mean()), 4)},
        },
        "total_mutations":        len(df_out),
        "total_bypasses":         total_bypass,
        "overall_bypass_rate":    round(float(all_bypass.mean()), 4),
        "top5_exploited_features":[f for f, _ in importance_ranked[:5]],
        "scale_min":              round(float(all_mutations.min()), 4),
        "scale_max":              round(float(all_mutations.max()), 4),
        "time_seconds":           round(elapsed, 1),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved         : {LOG_PATH}")
    print(f"\n  Next step: Run ewc.py")


if __name__ == "__main__":
    main()
