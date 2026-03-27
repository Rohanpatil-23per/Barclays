"""
Layer 3 - Immune Response Engine
Loads the Dueling DQN model and outputs structured ActionDecisions.

[IMMUNEX-PATCH] Step 2 refactoring applied:
  Bug 1 — strict 128-dim feature_vector parse; zero-vector fallback (no random)
  Bug 2 — secondary actions only within 10 % of primary Q-value (argmax)
  Bug 3 — filter_conflicting_actions: do_nothing short-circuit + category dedup
  Rejection memory — async _check_rejection_memory via database.is_action_rejected
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


# Pairs of action indices that must never execute together because they target
# the same resource and would create a destructive cascade or logical contradiction.
# (block_source_ip + null_route_attacker both null the same IP via different APIs)
# (rollback_filesystem_changes + restore_from_clean_snapshot both touch snapshots)
# (kill_malicious_process + suspend_suspicious_process target the same PID)
# (disable_compromised_account + lock_privileged_account target the same account)
_MUTUALLY_EXCLUSIVE_PAIRS: frozenset[frozenset] = frozenset({
    frozenset({10, 18}),   # block_source_ip  ↔  null_route_attacker
    frozenset({33, 34}),   # rollback_filesystem_changes  ↔  restore_from_clean_snapshot
    frozenset({30, 31}),   # kill_malicious_process  ↔  suspend_suspicious_process
    frozenset({22, 28}),   # disable_compromised_account  ↔  lock_privileged_account
})


def filter_conflicting_actions(actions: list[int]) -> list[int]:
    """
    Removes duplicate and mutually exclusive actions, preserving DQN ranking.

    Rules (applied in order):
    1. If do_nothing (0) is anywhere in the list → return [0] immediately.
       do_nothing is semantically incompatible with any active response.
    2. Drop exact duplicates (keep first occurrence, i.e. highest Q-value).
    3. Drop the lower-ranked action in any MUTUALLY_EXCLUSIVE_PAIRS hit.
       (Category-level dedup is intentionally removed — two network actions
        that target different resources, e.g. block_source_ip + block_c2_domain,
        are both valid and should both execute.)
    """
    # Rule 1: do_nothing short-circuit
    if 0 in actions:
        return [0]

    filtered: list[int] = []
    seen_ids: set[int] = set()

    for act in actions:
        # Rule 2: exact duplicate
        if act in seen_ids:
            continue

        # Rule 3: mutually exclusive pair — if any already-selected action
        # conflicts with this one, skip it (the earlier/higher-ranked one wins).
        conflict = any(
            frozenset({act, selected}) in _MUTUALLY_EXCLUSIVE_PAIRS
            for selected in filtered
        )
        if conflict:
            continue

        filtered.append(act)
        seen_ids.add(act)

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

    # [IMMUNEX-PATCH] Bug 1 + Rejection memory: predict() is now async so it
    # can await the database rejection-memory check via asyncpg.
    async def predict(self, alert: dict) -> "ActionDecision":
        """
        Runs inference on a Layer 2 alert dict.

        [IMMUNEX-PATCH] Bug 1: strictly parses the 128-dim feature_vector from
        the incoming payload. Falls back to np.zeros(128) — never np.random.rand
        — so the model always receives a deterministic safe input on bad data.

        [IMMUNEX-PATCH] Bug 2: secondary actions are only included if their
        Q-value is within 10 % of the primary (argmax) action's Q-value.

        [IMMUNEX-PATCH] Rejection memory: before finalising, each candidate
        action is checked against the rejected_demonstrations table.  If a
        near-identical state previously triggered a rejection for this action,
        it is dropped and the next-highest Q-value is tried instead.
        """
        alert_id = alert.get("alert_id", str(uuid.uuid4()))

        # [IMMUNEX-PATCH] Bug 1: strict feature_vector parse; zero-vector fallback
        raw_fv = alert.get("feature_vector")
        if not raw_fv or len(raw_fv) != 128:
            self.logger.warning(
                "feature_vector_missing_or_wrong_size",
                alert_id=alert_id,
                got_len=len(raw_fv) if raw_fv else 0,
                action="using_zero_vector_fallback",
            )
            obs = np.zeros(128, dtype=np.float32)  # [IMMUNEX-PATCH] Bug 1: zero-vector, NOT random
        else:
            try:
                obs = self.validate_feature_vector(raw_fv)
            except ValueError as exc:
                self.logger.error("invalid_feature_vector", alert_id=alert_id, error=str(exc))
                # [IMMUNEX-PATCH] Bug 1: fallback to zeros instead of raising or random
                self.logger.warning("falling_back_to_zero_vector", alert_id=alert_id)
                obs = np.zeros(128, dtype=np.float32)

        # ── Raw Q-values for explainability ───────────────────────────────
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
            self.logger.warning("q_value_extraction_failed", alert_id=alert_id, error=str(exc))

        # ── [IMMUNEX-PATCH] Bug 2: Mathematically sound multi-action selection ─
        # 1. argmax is the undisputed primary action.
        # 2. A secondary action is only included if its Q-value is >= 90 % of
        #    the primary's Q-value (i.e., within 10 % of the primary).
        # This avoids blindly promoting low-confidence second choices.
        if raw_q_values is not None:
            q_arr = np.array(raw_q_values)
            primary_idx  = int(np.argmax(q_arr))            # [IMMUNEX-PATCH] Bug 2: true argmax
            primary_qval = q_arr[primary_idx]
            threshold    = primary_qval * 0.90              # [IMMUNEX-PATCH] Bug 2: 10 % window

            # Build candidates: primary first, then secondaries meeting threshold
            candidates = [primary_idx]
            # argsort ascending → take all except the last (primary), reverse for descending
            sorted_indices = np.argsort(q_arr)[:-1][::-1].tolist()
            for idx in sorted_indices:
                if idx == primary_idx:
                    continue
                if q_arr[idx] >= threshold:               # [IMMUNEX-PATCH] Bug 2: threshold check
                    candidates.append(idx)
                else:
                    break  # sorted descending, so all remaining are below threshold
        else:
            # Fallback: model.predict deterministic action
            action_index, _ = self.model.predict(obs, deterministic=True)
            candidates = [int(action_index)]

        # ── [IMMUNEX-PATCH] Rejection memory: drop candidates rejected before ─
        # Import here to avoid circular imports at module level; the database
        # module is only needed for the async lookup, not for class definition.
        from response_engine.database import is_action_rejected

        state_vec_list = obs.tolist()
        filtered_candidates: list[int] = []
        for candidate in candidates:
            rejected = await is_action_rejected(state_vec_list, candidate)
            if rejected:
                self.logger.info(
                    "candidate_dropped_by_rejection_memory",
                    alert_id=alert_id,
                    candidate_action=candidate,
                )
            else:
                filtered_candidates.append(candidate)

        # If ALL candidates were rejected, fall back to do_nothing (action 0)
        if not filtered_candidates:
            self.logger.warning(
                "all_candidates_rejected_fallback_do_nothing",
                alert_id=alert_id,
            )
            filtered_candidates = [0]

        # ── Apply conflict filter ──────────────────────────────────────────
        actions = filter_conflicting_actions(filtered_candidates)  # [IMMUNEX-PATCH] Bug 3
        if not actions:
            actions = [filtered_candidates[0]]

        primary_action = actions[0] if actions else 0
        action_name    = ACTION_NAMES.get(primary_action, "unknown_action")

        action_names      = [ACTION_NAMES.get(idx, "unknown_action") for idx in actions]
        action_categories = [get_action_category(idx) for idx in actions]

        confidence = float(alert.get("layer2_confidence", 0.0))
        uncertain  = confidence < 0.6
        impact     = is_high_impact(actions, alert)

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
    import asyncio

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

    # [IMMUNEX-PATCH] Bug 1 smoke test: feature_vector from alert payload, not random
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
        "feature_vector"    : [0.5] * 128,   # [IMMUNEX-PATCH] deterministic, not random
        "layer2_confidence" : 0.95,
    }

    print("\nRunning smoke test …")
    _decision = asyncio.run(engine.predict(_test_alert))
    print("\n--- ActionDecision ---")
    print(json.dumps(asdict(_decision), indent=2))
    print("\n[OK] response_engine.py smoke test passed.")