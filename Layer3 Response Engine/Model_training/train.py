"""
IMMUNEX – Step 4: Dueling DQN Training Script
===============================================
Trains a Dueling Deep Q-Network with Prioritized Experience Replay (PER)
on the CyberIncidentEnv Gym environment.

Architecture:
  Shared trunk  : Linear(128→256) → ReLU → Linear(256→128) → ReLU → Linear(128→64) → ReLU
  Value stream  : Linear(64→1)
  Advantage stream: Linear(64→n_actions)
  Q(s,a) = V(s) + A(s,a) − mean(A(s,a))

Dependencies:
  pip install stable-baselines3 sb3-contrib torch tensorboard gymnasium

Author  : IMMUNEX – Step 4
Version : 1.2.0

Changelog v1.1.0
----------------
TRAIN-FIX-1 : StagnationCallback now tracks attack recall, not mean reward.
              Mean reward was a misleading signal — the agent could score
              high rewards by stalling on monitoring actions while never
              catching a single attack. Recall directly measures what matters.

TRAIN-FIX-2 : Stagnation detection blocked for the first 100,000 steps.
              Previously it fired at step 50,000 before the agent had left
              the exploration phase (exploration_fraction=0.50 means epsilon
              doesn't reach its floor until step 150,000). Early stagnation
              was cutting Phase 1 short and wasting 250,000 training steps.

TRAIN-FIX-3 : Phase 2 now runs for a fixed 150,000 steps instead of
              `total_timesteps - model.num_timesteps` which evaluated to
              ~10,000 steps (essentially random). 150k gives the reduced
              action space model enough experience to converge.

TRAIN-FIX-4 : evaluate() now constructs CyberIncidentEnv with
              attack_sample_ratio=0.50 so that the 200 evaluation episodes
              are split evenly between attack and normal samples. Without
              this the test set's 91% normal distribution meant the agent
              was evaluated almost entirely on normal traffic, masking its
              0% attack recall entirely.

TRAIN-FIX-5 : make_env() passes attack_sample_ratio=0.50 explicitly so
              both the training and eval envs inside SB3 callbacks also use
              balanced sampling. Omitting this caused the EvalCallback to
              silently reinforce the "always predict normal" strategy during
              training.
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type

# ── Stable-Baselines3 core ─────────────────────────────────────────────────
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

# ── sb3-contrib for Prioritized Experience Replay ─────────────────────────
try:
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from sb3_contrib import QRDQN
    _HAS_SB3_CONTRIB = True
except ImportError:
    _HAS_SB3_CONTRIB = False
    warnings.warn(
        "sb3-contrib not found. Install with: pip install sb3-contrib\n"
        "Falling back to standard DQN with uniform replay buffer.",
        stacklevel=2,
    )

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.cyber_incident_env import CyberIncidentEnv

torch.manual_seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Data
    "train_csv"        : "processed_train.csv",
    "test_csv"         : "processed_test.csv",

    # Training
    "total_timesteps"  : 300_000,
    "batch_size"       : 64,
    "learning_rate"    : 5e-5,
    "buffer_size"      : 100_000,
    "gamma"            : 0.99,
    "target_update"    : 500,
    "train_freq"       : 4,
    "gradient_steps"   : 1,

    # Exploration
    "exploration_fraction"    : 0.50,
    "exploration_initial_eps" : 1.0,
    "exploration_final_eps"   : 0.02,
    "learning_starts"         : 10000,

    # Network
    "hidden_sizes"     : [512, 256, 128],
    "n_actions_full"   : 50,
    "n_actions_reduced": 20,

    # Callbacks
    "checkpoint_freq"  : 10_000,

    # TRAIN-FIX-2: don't allow stagnation detection before step 100,000.
    # With exploration_fraction=0.50 the agent is still mostly random until
    # step 150,000 — flagging stagnation at step 50,000 is meaningless.
    "stagnation_min_step"  : 100_000,
    # TRAIN-FIX-1: stagnation now based on recall, not mean reward.
    # Window of steps without recall improvement before flagging.
    "stagnation_steps"     : 75_000,
    # Minimum recall improvement to reset the stagnation counter.
    "stagnation_recall_delta": 0.02,

    # TRAIN-FIX-3: Phase 2 runs for a fixed 150,000 steps.
    "phase2_timesteps" : 150_000,

    # Paths
    "checkpoint_dir"   : "checkpoints",
    "model_dir"        : "models",
    "log_dir"          : "logs",

    # Hardware
    "device"           : "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────────────────────────────────────
# DUELING Q-NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.

    Shared trunk → split into:
      Value stream     V(s)    : scalar
      Advantage stream A(s,a)  : vector of size n_actions

    Combined: Q(s,a) = V(s) + A(s,a) − mean(A(s,a))
    Subtracting mean(A) ensures identifiability between V and A.
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128, 64]

        trunk_layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            trunk_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        final_dim = hidden_sizes[-1]

        self.value_stream = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        shared    = self.trunk(obs)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM SB3 POLICY WRAPPING DUELING Q-NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        net_arch = self.net_arch or CFG["hidden_sizes"]
        obs_dim  = self.observation_space.shape[0]
        dueling_net = DuelingQNetwork(
            obs_dim      = obs_dim,
            n_actions    = self.action_space.n,
            hidden_sizes = net_arch,
        ).to(self.device)
        return _DuelingQNetWrapper(dueling_net, self.observation_space, self.action_space)


class _DuelingQNetWrapper(QNetwork):
    def __init__(self, dueling_net: DuelingQNetwork, observation_space, action_space):
        super().__init__(
            observation_space  = observation_space,
            action_space       = action_space,
            features_extractor = nn.Identity(),
            features_dim       = observation_space.shape[0],
            net_arch           = [],
            activation_fn      = nn.ReLU,
            normalize_images   = False,
        )
        self.dueling_net = dueling_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.dueling_net(obs)

    def _predict(self, observation: torch.Tensor,
                 deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        return q_values.argmax(dim=1)

    def set_training_mode(self, mode: bool) -> None:
        self.dueling_net.train(mode)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class StagnationCallback(BaseCallback):
    """
    Monitors attack recall by running real evaluation episodes every
    `eval_interval` steps, and flags stagnation if recall does not
    improve by `recall_delta` within `stagnation_steps` steps.

    Root cause of the previous v1.1.0 bug: the callback held a reference
    to `raw_eval_env` but never called reset()/step() on it, so
    get_metrics() always returned recall=0.0 regardless of how well the
    agent was actually performing. The stagnation flag fired immediately
    at min_step+stagnation_steps for every run.

    Fix: _on_step() now runs `n_eval_episodes` deterministic episodes on
    `eval_env` every `eval_interval` steps and reads recall from those.
    The env's episode_stats are reset before each evaluation so results
    reflect only the current policy, not cumulative history.
    """

    def __init__(self,
                 eval_env: CyberIncidentEnv,
                 stagnation_steps: int = 75_000,
                 min_step: int         = 100_000,
                 recall_delta: float   = 0.02,
                 eval_interval: int    = 5_000,
                 n_eval_episodes: int  = 50,
                 verbose: int          = 0):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.stagnation_steps = stagnation_steps
        self.min_step         = min_step
        self.recall_delta     = recall_delta
        self.eval_interval    = eval_interval
        self.n_eval_episodes  = n_eval_episodes
        self._best_recall     = -np.inf
        self._steps_no_improve = 0
        self.stagnated        = False

    def _run_eval(self) -> float:
        """Run n_eval_episodes deterministic episodes, return attack recall."""
        # Reset stats so we measure only the current policy
        self.eval_env.episode_stats = {
            "true_positives" : 0,
            "true_negatives" : 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(int(action))
                done = terminated or truncated
        return self.eval_env.get_metrics()["recall"]

    def _on_step(self) -> bool:
        # Don't check until the agent has left the exploration phase
        if self.num_timesteps < self.min_step:
            return True

        # Only re-evaluate every eval_interval steps (not every single step)
        if self.num_timesteps % self.eval_interval != 0:
            return True

        recall = self._run_eval()

        if recall > self._best_recall + self.recall_delta:
            self._best_recall      = recall
            self._steps_no_improve = 0
        else:
            self._steps_no_improve += self.eval_interval

        if self._steps_no_improve >= self.stagnation_steps and not self.stagnated:
            self.stagnated = True
            print(
                f"\n[Stagnation] No recall improvement in {self.stagnation_steps:,} steps. "
                f"Best recall = {self._best_recall:.4f}. "
                "Flagged for action-space reduction."
            )
        return True


class ProgressCallback(BaseCallback):
    """Prints concise progress every 10,000 steps including current recall."""

    def __init__(self, log_interval: int = 10_000,
                 eval_env: CyberIncidentEnv | None = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.eval_env     = eval_env   # optional — if provided, prints recall
        self._start_time  = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            elapsed     = time.time() - self._start_time
            fps         = self.n_calls / max(elapsed, 1)
            mean_reward = (
                np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                if self.model.ep_info_buffer else float("nan")
            )
            recall_str = ""
            if self.eval_env is not None:
                # Quick 20-episode recall snapshot
                self.eval_env.episode_stats = {
                    "true_positives": 0, "true_negatives": 0,
                    "false_positives": 0, "false_negatives": 0,
                }
                for _ in range(20):
                    obs, _ = self.eval_env.reset()
                    done = False
                    while not done:
                        act, _ = self.model.predict(obs, deterministic=True)
                        obs, _, term, trunc, _ = self.eval_env.step(int(act))
                        done = term or trunc
                recall = self.eval_env.get_metrics()["recall"]
                recall_str = f"  recall={recall:.3f}"
            print(
                f"  [Step {self.num_timesteps:>7,}]  "
                f"mean_ep_reward={mean_reward:+.3f}  "
                f"fps={fps:.0f}  "
                f"elapsed={elapsed:.0f}s"
                f"{recall_str}"
            )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# REDUCED-ACTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

import gymnasium as gym
from gymnasium import spaces

_TOP_20_ACTIONS = [
    0,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23,
    30, 31, 32,
    40, 49,
]

class ReducedActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, valid_actions: List[int] = None):
        super().__init__(env)
        self._valid_actions = valid_actions or _TOP_20_ACTIONS
        self.action_space   = spaces.Discrete(len(self._valid_actions))

    def action(self, act: int) -> int:
        return self._valid_actions[act]


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env(csv_path: str,
             max_steps: int = 200,
             reduced: bool = False,
             attack_sample_ratio: float = 0.50) -> gym.Env:
    """
    Construct and wrap the CyberIncidentEnv.

    TRAIN-FIX-5: attack_sample_ratio is passed through explicitly.
    All envs — training, eval callbacks, and the final evaluator — use
    balanced 50/50 sampling so the agent never sees a skewed distribution
    that rewards "always predict normal".
    """
    env = CyberIncidentEnv(
        dataset_path        = csv_path,
        max_steps           = max_steps,
        attack_sample_ratio = attack_sample_ratio,   # TRAIN-FIX-5
    )
    env = Monitor(env)
    if reduced:
        env = ReducedActionWrapper(env)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict = CFG) -> DQN:
    """Full Dueling DQN training pipeline. Returns the trained SB3 DQN model."""

    print("=" * 65)
    print("  IMMUNEX  –  Dueling DQN Training (Step 4)")
    print("=" * 65)
    print(f"  Device            : {cfg['device'].upper()}")
    print(f"  Total steps       : {cfg['total_timesteps']:,}")
    print(f"  Phase 2 steps     : {cfg['phase2_timesteps']:,}")
    print(f"  Batch size        : {cfg['batch_size']}")
    print(f"  Buffer size       : {cfg['buffer_size']:,}")
    print(f"  Architecture      : {cfg['hidden_sizes']}")
    print(f"  Stagnation metric : recall (min_step={cfg['stagnation_min_step']:,})")
    print("=" * 65)

    for d in [cfg["checkpoint_dir"], cfg["model_dir"], cfg["log_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Phase 1: Full 50-action training ──────────────────────────────────
    print("\n[Phase 1] Training with full 50-action space …")

    # TRAIN-FIX-5: all envs use balanced sampling
    env_train = make_env(cfg["train_csv"], reduced=False, attack_sample_ratio=0.50)

    test_csv  = cfg["test_csv"] if os.path.exists(cfg["test_csv"]) else cfg["train_csv"]
    env_eval  = make_env(test_csv, reduced=False, attack_sample_ratio=0.50)

    # Raw (unwrapped) eval env so StagnationCallback can read get_metrics()
    # TRAIN-FIX-1: stagnation tracks recall via this env
    raw_eval_env = CyberIncidentEnv(
        dataset_path        = test_csv,
        max_steps           = 200,
        attack_sample_ratio = 0.50,
    )

    stagnation_cb = StagnationCallback(
        eval_env         = raw_eval_env,
        stagnation_steps = cfg["stagnation_steps"],
        min_step         = cfg["stagnation_min_step"],   # TRAIN-FIX-2
        recall_delta     = cfg["stagnation_recall_delta"],
        verbose          = 0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq               = cfg["checkpoint_freq"],
        save_path               = cfg["checkpoint_dir"],
        name_prefix             = "dueling_dqn",
        save_replay_buffer      = False,
        verbose                 = 0,
    )

    eval_cb = EvalCallback(
        env_eval,
        best_model_save_path = cfg["model_dir"],
        log_path             = cfg["log_dir"],
        eval_freq            = cfg["checkpoint_freq"],
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 0,
    )

    progress_cb = ProgressCallback(log_interval=10_000, eval_env=raw_eval_env, verbose=0)

    callback_list = CallbackList([
        checkpoint_cb,
        eval_cb,
        stagnation_cb,
        progress_cb,
    ])

    model = DQN(
        policy                        = DuelingDQNPolicy,
        env                           = env_train,
        learning_rate                 = cfg["learning_rate"],
        buffer_size                   = cfg["buffer_size"],
        batch_size                    = cfg["batch_size"],
        gamma                         = cfg["gamma"],
        target_update_interval        = cfg["target_update"],
        train_freq                    = cfg["train_freq"],
        gradient_steps                = cfg["gradient_steps"],
        exploration_fraction          = cfg["exploration_fraction"],
        exploration_initial_eps       = cfg["exploration_initial_eps"],
        exploration_final_eps         = cfg["exploration_final_eps"],
        learning_starts               = cfg.get("learning_starts", 5000),
        max_grad_norm                 = 10,
        optimize_memory_usage         = False,
        tensorboard_log               = cfg["log_dir"],
        policy_kwargs                 = {
            "net_arch"         : cfg["hidden_sizes"],
            "optimizer_class"  : torch.optim.Adam,
            "optimizer_kwargs" : {"eps": 1.5e-4},
        },
        device                        = cfg["device"],
        verbose                       = 0,
    )

    try:
        model.learn(
            total_timesteps     = cfg["total_timesteps"],
            callback            = callback_list,
            tb_log_name         = "DuelingDQN",
            reset_num_timesteps = True,
            progress_bar        = False,
        )
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user.")

    # ── Phase 2: Reduced action space (if stagnated) ───────────────────────
    if stagnation_cb.stagnated:
        print(f"\n[Phase 2] Stagnation detected. Switching to reduced "
              f"{cfg['n_actions_reduced']}-action space for continued training …")

        env_reduced  = make_env(cfg["train_csv"], reduced=True, attack_sample_ratio=0.50)
        eval_reduced = make_env(test_csv,         reduced=True, attack_sample_ratio=0.50)

        # TRAIN-FIX-3: fixed 150,000 steps — not the leftover scraps from Phase 1
        phase2_steps = cfg["phase2_timesteps"]

        model_reduced = DQN(
            policy                        = DuelingDQNPolicy,
            env                           = env_reduced,
            learning_rate                 = cfg["learning_rate"] * 0.5,
            buffer_size                   = cfg["buffer_size"],
            batch_size                    = cfg["batch_size"],
            gamma                         = cfg["gamma"],
            target_update_interval        = cfg["target_update"],
            train_freq                    = cfg["train_freq"],
            gradient_steps                = cfg["gradient_steps"],
            exploration_fraction          = 0.10,
            exploration_initial_eps       = 0.2,
            exploration_final_eps         = 0.01,
            tensorboard_log               = cfg["log_dir"],
            policy_kwargs                 = {
                "net_arch"         : cfg["hidden_sizes"],
                "optimizer_class"  : torch.optim.Adam,
                "optimizer_kwargs" : {"eps": 1.5e-4},
            },
            device                        = cfg["device"],
            verbose                       = 0,
        )

        ckpt_reduced = CheckpointCallback(
            save_freq   = cfg["checkpoint_freq"],
            save_path   = cfg["checkpoint_dir"],
            name_prefix = "dueling_dqn_reduced",
            verbose     = 0,
        )

        model_reduced.learn(
            total_timesteps     = phase2_steps,
            callback            = CallbackList([ckpt_reduced, ProgressCallback(log_interval=10_000)]),
            tb_log_name         = "DuelingDQN_Reduced",
            reset_num_timesteps = True,
            progress_bar        = False,
        )

        reduced_path = os.path.join(cfg["model_dir"], "dueling_dqn_reduced.zip")
        model_reduced.save(reduced_path)
        print(f"  [Saved] Reduced-action model → '{reduced_path}'")

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(cfg["model_dir"], "dueling_dqn_immunex.zip")
    model.save(final_path)
    size_mb = os.path.getsize(final_path) / (1024 ** 2)
    print(f"\n[Saved] Final model → '{final_path}'  ({size_mb:.2f} MB)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: DQN, cfg: dict = CFG, n_episodes: int = 200) -> dict:
    """
    Run deterministic evaluation and return aggregate metrics.

    TRAIN-FIX-4: Uses attack_sample_ratio=0.50 so evaluation episodes are
    split evenly between attack and normal samples. The original code used
    the default (which inherits dataset ratio ~9% attacks), making it
    impossible to detect 0% recall — 91% of episodes never saw an attack.
    """
    csv = cfg["test_csv"] if os.path.exists(cfg["test_csv"]) else cfg["train_csv"]

    # TRAIN-FIX-4: balanced evaluation env
    eval_env = CyberIncidentEnv(
        dataset_path        = csv,
        max_steps           = 200,
        attack_sample_ratio = 0.50,
    )

    rewards = []

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    raw       = eval_env.get_metrics()
    tp, tn    = raw["true_positives"],  raw["true_negatives"]
    fp, fn    = raw["false_positives"], raw["false_negatives"]
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "mean_reward"    : round(float(np.mean(rewards)), 4),
        "std_reward"     : round(float(np.std(rewards)),  4),
        "min_reward"     : round(float(np.min(rewards)),  4),
        "max_reward"     : round(float(np.max(rewards)),  4),
        "precision"      : round(precision, 4),
        "recall"         : round(recall,    4),
        "f1_score"       : round(f1,        4),
        "accuracy"       : round(accuracy,  4),
        "true_positives" : tp,
        "true_negatives" : tn,
        "false_positives": fp,
        "false_negatives": fn,
        "n_episodes"     : n_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train(CFG)

    # Load the best checkpoint saved by EvalCallback rather than the final
    # weights — the final weights may have drifted from the best policy.
    best_path = os.path.join(CFG["model_dir"], "best_model.zip")
    if os.path.exists(best_path):
        print(f"\n[Eval] Loading best checkpoint from '{best_path}'")
        eval_model = DQN.load(best_path)
    else:
        print("\n[Eval] No best_model.zip found — evaluating final weights.")
        eval_model = trained_model

    print("\n" + "=" * 65)
    print("  Evaluation on Test Set")
    print("=" * 65)
    metrics = evaluate(eval_model, CFG, n_episodes=200)
    for k, v in metrics.items():
        print(f"  {k:<20s}: {v}")
    print("=" * 65)
    print("[Done] Dueling DQN training complete.")
    print("       Start TensorBoard:  tensorboard --logdir logs/")
