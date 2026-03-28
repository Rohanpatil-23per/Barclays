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