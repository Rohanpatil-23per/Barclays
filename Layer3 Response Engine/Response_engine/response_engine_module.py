"""
Layer 3 - Immune Response Engine
Loads the Dueling DQN model and outputs structured ActionDecisions.
"""

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
import structlog
from stable_baselines3 import DQN

from action_registry import ACTION_NAMES, get_action_category

# ── Structlog configuration ────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# ── User-Defined Risk Logic ───────────────────────────────────────────────

def is_high_impact(actions: list[int], alert: dict) -> str:
    """
    Dynamically determine the impact level based on severity and actions.
    Returns "high", "medium", or "low".
    """
    severity = alert.get("severity", "low").lower()
    if severity in ("critical", "high"):
        return "high"
    
    # Check if any action is typically high impact (network/process/data containment)
    for act in actions:
        cat = get_action_category(act)
        if cat in ("network", "process", "data_protection"):
            return "high"
            
    return "medium" if severity == "medium" else "low"


def filter_conflicting_actions(actions: list[int]) -> list[int]:
    """
    Removes duplicate and mutually exclusive actions, preserving DQN ranking.
    Keeps the highest-priority actions.
    """
    filtered = []
    seen = set()
    
    for act in actions:
        if act in seen:
            continue
            
        # Basic conflict rule 1: Do not mix 'do_nothing' (assumed index 0) with active interventions.
        if act == 0 and len(filtered) > 0:
            continue
        if 0 in filtered:
            continue
            
        # Additional mutually exclusive constraints can be mapped here using ACTION_NAMES.
        
        filtered.append(act)
        seen.add(act)
        
    return filtered


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class ActionDecision:
    alert_id: str
    action_index: int | None
    action_name: str | None
    actions: list[int]
    action_names: list[str]
    action_categories: list[str]
    requires_approval: bool
    confidence: float
    uncertain: bool
    impact: str
    severity: str
    timestamp: str
    raw_q_values: list[float] | None = None


# ── Engine ─────────────────────────────────────────────────────────────────

class ResponseEngine:
    """
    Loads a trained Dueling DQN model and produces ActionDecisions from
    Layer 2 alert dicts.
    """

    def __init__(self, model_path: str):
        self.logger = structlog.get_logger("response_engine")

        if not os.path.exists(model_path):
            msg = f"Model file not found: {model_path}"
            self.logger.error("model_load_failed", path=model_path)
            raise RuntimeError(msg)

        try:
            self.logger.info("loading_model", path=model_path)
            self.model_path = model_path
            self.model = DQN.load(model_path)
            self.logger.info("model_loaded_successfully", path=model_path)
        except Exception as exc:
            self.logger.error("model_load_exception", path=model_path, error=str(exc))
            raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_feature_vector(self, feature_vector: list) -> np.ndarray:
        """
        Validates shape and content of the incoming feature vector.

        Returns a float32 numpy array.
        Raises ValueError with a descriptive message on any problem.
        """
        if not isinstance(feature_vector, (list, np.ndarray)):
            raise ValueError(
                f"feature_vector must be a list or ndarray, got {type(feature_vector).__name__}"
            )

        if len(feature_vector) == 0:
            raise ValueError("feature_vector must not be empty")

        # Explicit float32 cast — input may arrive as float64 from numpy or JSON
        arr = np.array(feature_vector, dtype=np.float32)
        # RL model trained on 128-dim — truncate or zero-pad to match
        TARGET = 128
        if arr.shape[0] > TARGET:
            arr = arr[:TARGET]
        elif arr.shape[0] < TARGET:
            arr = np.concatenate([arr, np.zeros(TARGET - arr.shape[0], dtype=np.float32)])

        if np.isnan(arr).any():
            raise ValueError("feature_vector contains NaN values")

        if np.isinf(arr).any():
            raise ValueError("feature_vector contains Inf values")

        return arr

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, alert: dict) -> ActionDecision:
        """
        Runs inference on a Layer 2 alert dict.

        Extracts the feature vector, runs the DQN, and returns a
        fully-populated ActionDecision.
        """
        alert_id = alert.get("alert_id", str(uuid.uuid4()))

        try:
            obs = self.validate_feature_vector(alert.get("feature_vector", []))
        except ValueError as exc:
            self.logger.error("invalid_feature_vector",
                              alert_id=alert_id, error=str(exc))
            raise

        # ── Raw Q-values for explainability ───────────────────────────────
        # SB3 wraps the network inside policy.q_net — NOT model.q_net directly.
        # Using policy.q_net(obs_tensor) is the correct path in SB3 >= 1.6.
        raw_q_values: list[float] | None = None
        try:
            with torch.no_grad():
                obs_tensor = (
                    torch.as_tensor(obs)
                    .float()
                    .unsqueeze(0)
                    .to(self.model.device)
                )
                # FIX: self.model.policy.q_net, not self.model.q_net
                q_values = self.model.policy.q_net(obs_tensor)
                raw_q_values = q_values.cpu().numpy()[0].tolist()
        except Exception as exc:
            # Q-value extraction is optional — don't abort if it fails
            self.logger.warning("q_value_extraction_failed",
                                alert_id=alert_id, error=str(exc))

        # ── Multi-action selection ─────────────────────────────────────────
        if raw_q_values is not None:
            k = 3  # Select top 3 actions
            # argsort returns ascending, so take last k and reverse
            top_indices = np.argsort(raw_q_values)[-k:][::-1].tolist()
        else:
            # Fallback if Q-value extraction failed
            action_index, _ = self.model.predict(obs, deterministic=True)
            top_indices = [int(action_index)]

        # Filter duplicates and conflicts, maintaining DQN ranking
        actions = filter_conflicting_actions(top_indices)
        
        if not actions:
            actions = [top_indices[0]]

        primary_action = actions[0] if actions else 0
        action_name = ACTION_NAMES.get(primary_action, "unknown_action")
        
        action_names = [ACTION_NAMES.get(idx, "unknown_action") for idx in actions]
        action_categories = [get_action_category(idx) for idx in actions]
        
        confidence = float(alert.get("layer2_confidence", 0.0))
        uncertain = confidence < 0.6
        impact = is_high_impact(actions, alert)

        decision = ActionDecision(
            alert_id          = alert_id,
            action_index      = primary_action,
            action_name       = action_name,
            actions           = actions,
            action_names      = action_names,
            action_categories = action_categories,
            requires_approval = True,
            confidence        = confidence,
            uncertain         = uncertain,
            impact            = impact,
            severity          = alert.get("severity", "unknown"),
            timestamp         = datetime.utcnow().isoformat() + "Z",
            raw_q_values      = raw_q_values,
        )

        self.logger.info(
            "action_decision_made",
            alert_id          = decision.alert_id,
            primary_action    = decision.action_name,
            actions           = decision.actions,
            requires_approval = decision.requires_approval,
            uncertain         = decision.uncertain,
            impact            = decision.impact,
            severity          = decision.severity,
        )

        return decision

    # ── Online Retraining ──────────────────────────────────────────────────

    def retrain(
        self,
        expert_data: list[dict],
        rejected_data: list[dict],
        epochs: int = 5,
        lr: float = 1e-4,
        neg_weight: float = 1.0
    ) -> dict:
        """
        Performs Behavioral Cloning (BC) on expert demonstrations and penalises
        rejected demonstrations.
        """
        if not expert_data and not rejected_data:
            return {"status": "no_data", "samples": 0}

        self.logger.info("retraining_started",
                         expert_samples=len(expert_data),
                         rejected_samples=len(rejected_data))

        # We need the Q-network and optimizer
        q_net = self.model.policy.q_net
        optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

        # Prepare positive data
        states_pos, actions_pos = [], []
        for d in expert_data:
            try:
                state = self.validate_feature_vector(d["state_vector"])
                states_pos.append(state)
                actions_pos.append(int(d["expert_action"]))
            except Exception:
                pass

        # Prepare negative data
        states_neg, actions_neg = [], []
        for d in rejected_data:
            try:
                state = self.validate_feature_vector(d["state_vector"])
                states_neg.append(state)
                actions_neg.append(int(d["rejected_action"]))
            except Exception:
                pass

        if not states_pos and not states_neg:
            return {"status": "failed", "reason": "No valid data after parsing"}

        # Convert to tensors
        device = self.model.device
        
        t_states_pos = torch.as_tensor(np.array(states_pos)).float().to(device) if states_pos else None
        t_actions_pos = torch.as_tensor(actions_pos).long().to(device) if actions_pos else None
        
        t_states_neg = torch.as_tensor(np.array(states_neg)).float().to(device) if states_neg else None
        t_actions_neg = torch.as_tensor(actions_neg).long().to(device) if actions_neg else None

        q_net.train()
        total_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.tensor(0.0).to(device)
            
            # Behavioral cloning loss (CrossEntropy)
            if t_states_pos is not None:
                q_values_pos = q_net(t_states_pos)
                loss_pos = F.cross_entropy(q_values_pos, t_actions_pos)
                loss += loss_pos
                
            # Negative sampling loss (minimize probability of rejected action)
            if t_states_neg is not None:
                q_values_neg = q_net(t_states_neg)
                probs_neg = torch.softmax(q_values_neg, dim=-1)
                prob_rejected = probs_neg.gather(1, t_actions_neg.unsqueeze(1)).squeeze(1)
                loss_neg = prob_rejected.mean() * neg_weight
                loss += loss_neg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        q_net.eval()
        avg_loss = total_loss / max(1, epochs)
        
        # Save model using SB3's built in mechanism
        # Find the path it was originally loaded from, or a default
        try:
            self.model.save(self.model_path)
            self.logger.info("retraining_completed",
                             loss=avg_loss,
                             saved_path=self.model_path)
            saved = True
        except Exception as e:
            self.logger.error("retraining_save_failed", error=str(e))
            saved = False

        return {
            "status": "success",
            "expert_samples": len(states_pos),
            "rejected_samples": len(states_neg),
            "epochs": epochs,
            "final_loss": float(avg_loss),
            "saved": saved
        }


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Resolve model path relative to this file, then fall back to project root
    _here       = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_here, "models", "dueling_dqn_immunex.zip"),
        os.path.join(os.path.dirname(_here), "models", "dueling_dqn_immunex.zip"),
    ]
    _model_path = next((p for p in _candidates if os.path.exists(p)), None)

    if _model_path is None:
        print("ERROR: dueling_dqn_immunex.zip not found in expected locations:")
        for p in _candidates:
            print(f"  {p}")
        sys.exit(1)

    engine = ResponseEngine(model_path=_model_path)

    # np.random.rand produces float64 — validate_feature_vector casts to float32
    _test_alert = {
        "alert_id"          : str(uuid.uuid4()),
        "timestamp"         : datetime.utcnow().isoformat() + "Z",
        "source_ip"         : "192.168.1.100",
        "destination_ip"    : "10.0.0.50",
        "source_port"       : 4444,
        "destination_port"  : 443,
        "protocol"          : "TCP",
        "severity"          : "high",
        "attack_type"       : "C2_Beacon",
        "feature_vector"    : np.random.rand(128).tolist(),
        "layer2_confidence" : 0.95,
    }

    print("\nRunning smoke test …")
    _decision = engine.predict(_test_alert)
    print("\n--- ActionDecision ---")
    print(json.dumps(asdict(_decision), indent=2))
    print("\n[OK] response_engine.py smoke test passed.")