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

from response_engine.action_registry import ACTION_NAMES, get_action_category

# ── Structlog configuration ────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# Actions that require explicit human approval before execution
HIGH_IMPACT_ACTIONS: list[int] = [12, 13, 14, 18, 22, 28, 30, 32, 33, 34, 46]


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class ActionDecision:
    alert_id: str
    action_index: int
    action_name: str
    action_category: str
    requires_approval: bool
    confidence: float
    severity: str
    timestamp: str
    raw_q_values: list[float] | None = None


# ── Engine ─────────────────────────────────────────────────────────────────

class ResponseEngine:
    """
    Loads a trained Dueling DQN model and produces ActionDecisions from
    Layer 2 alert dicts.
    """

    def __init__(self, model_path: str,
                 require_approval_for: list[int] = HIGH_IMPACT_ACTIONS):
        self.logger = structlog.get_logger("response_engine")
        self.require_approval_for = require_approval_for

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

        Returns a (128,) float32 numpy array.
        Raises ValueError with a descriptive message on any problem.
        """
        if not isinstance(feature_vector, (list, np.ndarray)):
            raise ValueError(
                f"feature_vector must be a list or ndarray, got {type(feature_vector).__name__}"
            )

        if len(feature_vector) != 128:
            raise ValueError(
                f"feature_vector must have exactly 128 elements, got {len(feature_vector)}"
            )

        # Explicit float32 cast — input may arrive as float64 from numpy or JSON
        arr = np.array(feature_vector, dtype=np.float32)

        if np.isnan(arr).any():
            raise ValueError("feature_vector contains NaN values")

        if np.isinf(arr).any():
            raise ValueError("feature_vector contains Inf values")

        return arr

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, alert: dict) -> ActionDecision:
        """
        Runs inference on a Layer 2 alert dict.

        Extracts the 128-dim feature vector, runs the DQN, and returns a
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

        # ── Greedy action selection ────────────────────────────────────────
        action_index, _ = self.model.predict(obs, deterministic=True)
        action_index = int(action_index)

        action_name     = ACTION_NAMES.get(action_index, "unknown_action")
        action_category = get_action_category(action_index)
        requires_approval = action_index in self.require_approval_for

        decision = ActionDecision(
            alert_id          = alert_id,
            action_index      = action_index,
            action_name       = action_name,
            action_category   = action_category,
            requires_approval = requires_approval,
            confidence        = float(alert.get("layer2_confidence", 0.0)),
            severity          = alert.get("severity", "unknown"),
            timestamp         = datetime.utcnow().isoformat() + "Z",
            raw_q_values      = raw_q_values,
        )

        self.logger.info(
            "action_decision_made",
            alert_id          = decision.alert_id,
            action_index      = decision.action_index,
            action_name       = decision.action_name,
            action_category   = decision.action_category,
            requires_approval = decision.requires_approval,
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

    engine = ResponseEngine(model_path=_model_path,
                            require_approval_for=HIGH_IMPACT_ACTIONS)

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
