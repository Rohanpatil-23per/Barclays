"""
IMMUNEX — Layer 3: Z3 Safety Verifier
======================================
Formally verifies every ActionDecision against operational and compliance
constraints BEFORE execution.  Sits between ResponseEngine (DQN inference)
and the ActionExecutor.

Constraints encoded
-------------------
  C1  Trading-window protection     (actions 13, 14, 46 → IST 09:00–17:00)
  C2  Severity floor                (containment ≥ action 10 needs high/critical)
  C3  Self-IP protection            (never block management IP)
  C4  Rate-limit per source IP      (≤ 5 network blocks in 60 s)
  C5  Cascade prevention            (rollback/restore needs backup registry)
  C6  Credential audit requirement  (actions 20–29 always audited)

[IMMUNEX-PATCH] Step 3 changes:
  Bug 4 — C1 now uses explicit is_human_reviewed bool (not decision.requires_approval)
  Bug 5 — C3 checks action_params.get("target_ip") for firewall-block actions
  Bug 6 — Rate limiter uses request_arrival_time param, not time.monotonic() at exec
  Multi-action — verify() accepts actions list; any() loop blocks if ANY action violates
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque

import pytz
import structlog
from z3 import Bool, And, Not, Or, Solver, sat, unsat

from response_engine.response_engine import ActionDecision
from response_engine.action_registry import ACTION_NAMES, get_action_category

# ── Logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

_IST = pytz.timezone("Asia/Kolkata")

# ── Severity ordering ──────────────────────────────────────────────────────
_SEVERITY_RANK: dict[str, int] = {
    "low":      0,
    "medium":   1,
    "high":     2,
    "critical": 3,
}

# ── Constraint names ───────────────────────────────────────────────────────
C1 = "C1_trading_window_protection"
C2 = "C2_severity_floor"
C3 = "C3_self_ip_protection"
C4 = "C4_rate_limit"
C5 = "C5_cascade_prevention"
C6 = "C6_credential_audit"


# ── VerificationResult ─────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    approved:                bool
    violated_constraints:    list[str]
    substituted_action:      ActionDecision | None
    requires_audit_log:      bool
    requires_human_approval: bool
    reason:                  str


# ── SafetyVerifier ─────────────────────────────────────────────────────────

class SafetyVerifier:
    """
    Encodes six operational/compliance constraints as Z3 Boolean assertions
    and evaluates them for every incoming ActionDecision.

    Parameters
    ----------
    mgmt_ip              : str   IP of IMMUNEX management interface.
                                 Overridden by IMMUNEX_MGMT_IP env var.
    backup_registry_path : str   Path to JSON file listing available snapshot
                                 identifiers.
                                 Overridden by IMMUNEX_BACKUP_REGISTRY env var.
    """

    # Actions guarded by trading-window (C1)
    _TRADING_WINDOW_ACTIONS: frozenset[int] = frozenset({13, 14, 46})

    # Actions that must never target the management IP (C3)
    _IP_BLOCK_ACTIONS: frozenset[int] = frozenset({10, 11, 12, 13, 18})

    # Network-blocking actions counted by rate limiter (C4)
    _NETWORK_BLOCK_ACTIONS: frozenset[int] = frozenset(range(10, 20))

    # Restore/rollback actions that need a backup (C5)
    _RESTORE_ACTIONS: frozenset[int] = frozenset({33, 34})

    # Credential actions that always require audit (C6)
    _CREDENTIAL_ACTIONS: frozenset[int] = frozenset(range(20, 30))

    # Rate-limit window parameters
    _RATE_WINDOW_SECS: int = 60
    _RATE_MAX_HITS:    int = 5

    def __init__(
        self,
        mgmt_ip:              str = "127.0.0.1",
        backup_registry_path: str = "",
    ) -> None:
        self.logger = structlog.get_logger("safety_verifier")

        # 12-factor: environment variables override constructor defaults
        self.mgmt_ip: str = os.environ.get("IMMUNEX_MGMT_IP", mgmt_ip)
        self.backup_registry_path: str = os.environ.get(
            "IMMUNEX_BACKUP_REGISTRY", backup_registry_path
        )

        # [IMMUNEX-PATCH] Bug 6: Sliding-window rate limiter keyed by source_ip.
        # Values are deques of request_arrival_time floats (NOT time.monotonic()
        # at execution time) so queue-wait time cannot bypass the rate window.
        self._rate_window: dict[str, Deque[float]] = {}

        self.logger.info(
            "safety_verifier_initialised",
            mgmt_ip          = self.mgmt_ip,
            backup_registry  = self.backup_registry_path or "<not set>",
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def verify(
        self,
        decision:            ActionDecision,
        alert:               dict,
        # [IMMUNEX-PATCH] Bug 4: explicit is_human_reviewed boolean.
        # Previously the verifier used `decision.requires_approval` which always
        # starts as True (set by the DQN), so C1 could never be satisfied for
        # trading-window actions — a human override would still be blocked.
        is_human_reviewed:   bool = False,
        # [IMMUNEX-PATCH] Bug 6: request_arrival_time from the FastAPI endpoint
        # entry point (time.perf_counter() at first line of /respond or /approve).
        # Using this prevents queue-wait duration from eating into the rate window.
        request_arrival_time: float | None = None,
        # [IMMUNEX-PATCH] Bug 5: action_params carries the intended target IP
        # for firewall-block actions so C3 checks the right address.
        action_params:        dict | None = None,
    ) -> VerificationResult:
        """
        [IMMUNEX-PATCH] Multi-action verify: iterates over decision.actions using
        any() semantics — if ANY action in the multi-step array violates a rule,
        the entire batch is blocked.

        Returns a VerificationResult indicating whether the action is
        approved, which constraints were violated, and whether a substitute
        action has been provided.

        FIX: _record_action() is called here (and ONLY here) when an action
        is approved.  The smoke test must NOT call _record_action() manually
        alongside verify() — doing so caused double-counting and made the
        rate-limit test unreliable.
        """
        # [IMMUNEX-PATCH] Bug 6: fall back to current monotonic if caller
        # did not supply request_arrival_time (e.g. smoke tests).
        arrival_time = request_arrival_time if request_arrival_time is not None \
                       else time.monotonic()

        action_params = action_params or {}
        alert_id    = decision.alert_id
        severity    = decision.severity
        source_ip   = alert.get("source_ip", "")
        dest_ip     = alert.get("destination_ip", "")
        timestamp   = decision.timestamp

        # [IMMUNEX-PATCH] Multi-action: iterate over the full action list.
        # We run the Z3 check once per action; the first violation short-circuits.
        # Per-action results are accumulated into a single VerificationResult.
        violated:       list[str]            = []
        reasons:        list[str]            = []
        substituted:    ActionDecision | None = None
        requires_audit  = False
        requires_human  = False  # set True only when a constraint explicitly demands it
        approved        = True               # assume OK until a violation is found

        for action_idx in decision.actions:
            action_name = ACTION_NAMES.get(action_idx, f"unknown_{action_idx}")

            # ── Pre-compute Python-side predicate values ───────────────────
            in_trading_hours   = self._is_trading_hours(timestamp)
            is_trading_guarded = action_idx in self._TRADING_WINDOW_ACTIONS

            severity_rank  = _SEVERITY_RANK.get(severity, 0)
            needs_high_sev = action_idx >= 10
            has_high_sev   = severity_rank >= _SEVERITY_RANK["high"]

            # [IMMUNEX-PATCH] Bug 5: check action_params.get("target_ip") first
            # for firewall-block actions so C3 checks the actual intended target,
            # not just the alert's source_ip/dest_ip which may differ.
            target_ip = action_params.get("target_ip", "")
            effective_ips = {source_ip, dest_ip}
            if target_ip:
                effective_ips.add(target_ip)  # [IMMUNEX-PATCH] Bug 5
            mgmt_ip_hit        = any(ip == self.mgmt_ip for ip in effective_ips)
            is_ip_block_action = action_idx in self._IP_BLOCK_ACTIONS

            is_network_block = action_idx in self._NETWORK_BLOCK_ACTIONS
            # [IMMUNEX-PATCH] Bug 6: pass arrival_time into _check_rate_limit
            rate_ok          = self._check_rate_limit(source_ip, action_idx, arrival_time)

            is_restore_action = action_idx in self._RESTORE_ACTIONS
            backup_ok         = self._backup_exists() if is_restore_action else True

            is_credential_action = action_idx in self._CREDENTIAL_ACTIONS

            # ── Build Z3 solver ────────────────────────────────────────────
            solver = Solver()
            solver.set("timeout", 500)   # 500 ms — default to DENY on timeout

            # One Z3 Bool per constraint (True = constraint is satisfied)
            z3_c1 = Bool(f"c1_trading_window_{action_idx}")
            z3_c2 = Bool(f"c2_severity_floor_{action_idx}")
            z3_c3 = Bool(f"c3_self_ip_{action_idx}")
            z3_c4 = Bool(f"c4_rate_limit_{action_idx}")
            z3_c5 = Bool(f"c5_cascade_{action_idx}")
            z3_c6 = Bool(f"c6_credential_audit_{action_idx}")

            # [IMMUNEX-PATCH] Bug 4: C1 now uses is_human_reviewed (the explicit
            # boolean the caller passes in) rather than decision.requires_approval.
            # This ensures that actions 13 and 14 during market hours are ONLY
            # unblocked when a human admin has physically confirmed the override —
            # not simply because the DQN set requires_approval=True by default.
            z3_human = Bool(f"human_reviewed_{action_idx}")
            solver.add(z3_human == is_human_reviewed)  # [IMMUNEX-PATCH] Bug 4
            solver.add(
                z3_c1 == Or(
                    Not(And(in_trading_hours, is_trading_guarded)),
                    z3_human,   # [IMMUNEX-PATCH] Bug 4: gate on actual human review
                )
            )

            # C2: safe if action < 10 (monitoring) OR severity is high/critical
            solver.add(
                z3_c2 == Or(Not(needs_high_sev), has_high_sev)
            )

            # C3: safe if NOT (mgmt IP hit AND this is a blocking action)
            solver.add(
                z3_c3 == Not(And(mgmt_ip_hit, is_ip_block_action))  # [IMMUNEX-PATCH] Bug 5
            )

            # C4: safe if NOT a network block OR rate limit not exceeded
            solver.add(
                z3_c4 == Or(Not(is_network_block), rate_ok)
            )

            # C5: safe if NOT a restore action OR a backup exists
            solver.add(
                z3_c5 == Or(Not(is_restore_action), backup_ok)
            )

            # C6: credential actions are never blocked — always True
            solver.add(z3_c6 == True)  # noqa: E712 — Z3 requires the literal

            # All six must hold simultaneously
            solver.add(And(z3_c1, z3_c2, z3_c3, z3_c4, z3_c5, z3_c6))

            # ── Run solver ─────────────────────────────────────────────────
            result = solver.check()

            if result == sat:
                act_approved = True

            elif result == unsat:
                act_approved = False
                approved = False  # [IMMUNEX-PATCH] Multi-action: any violation → block all

                # Identify which specific constraints failed for this action
                if in_trading_hours and is_trading_guarded and not is_human_reviewed:
                    # [IMMUNEX-PATCH] Bug 4: key diagnostic phrase updated
                    violated.append(C1)
                    requires_human = True
                    reasons.append(
                        f"{action_name} is restricted during trading hours "
                        "(09:00–17:00 IST) — requires explicit human review via override"
                    )
                    self.logger.warning(
                        "constraint_violated",
                        constraint_name = C1,
                        action_name     = action_name,
                        alert_id        = alert_id,
                        is_human_reviewed = is_human_reviewed,
                        reason          = reasons[-1],
                    )

                if needs_high_sev and not has_high_sev:
                    violated.append(C2)
                    reasons.append(
                        f"{action_name} (index {action_idx}) requires severity "
                        f"high/critical, got '{severity}'"
                    )
                    self.logger.warning(
                        "constraint_violated",
                        constraint_name = C2,
                        action_name     = action_name,
                        alert_id        = alert_id,
                        reason          = reasons[-1],
                    )

                if mgmt_ip_hit and is_ip_block_action:
                    violated.append(C3)
                    # [IMMUNEX-PATCH] Bug 5: include target_ip in log
                    reasons.append(
                        f"{action_name} would target management IP {self.mgmt_ip} "
                        f"(source={source_ip}, dest={dest_ip}, target_ip={target_ip or 'N/A'})"
                    )
                    self.logger.warning(
                        "constraint_violated",
                        constraint_name = C3,
                        action_name     = action_name,
                        alert_id        = alert_id,
                        target_ip       = target_ip or "N/A",
                        reason          = reasons[-1],
                    )

                if is_network_block and not rate_ok:
                    violated.append(C4)
                    reasons.append(
                        f"Rate limit exceeded: {source_ip} has ≥{self._RATE_MAX_HITS} "
                        f"network-blocking actions in the last {self._RATE_WINDOW_SECS}s"
                    )
                    self.logger.warning(
                        "constraint_violated",
                        constraint_name = C4,
                        action_name     = action_name,
                        alert_id        = alert_id,
                        reason          = reasons[-1],
                    )

                if is_restore_action and not backup_ok:
                    violated.append(C5)
                    reasons.append(
                        f"{action_name} requires a verified backup snapshot but none "
                        f"found in '{self.backup_registry_path}'. "
                        "Substituting backup_critical_data (action 47)."
                    )
                    self.logger.warning(
                        "constraint_violated",
                        constraint_name = C5,
                        action_name     = action_name,
                        alert_id        = alert_id,
                        reason          = reasons[-1],
                    )
                    substituted = ActionDecision(
                        alert_id          = decision.alert_id,
                        action_index      = 47,
                        action_name       = ACTION_NAMES[47],
                        actions           = [47],
                        action_names      = [ACTION_NAMES[47]],
                        action_categories = [get_action_category(47)],
                        requires_approval = False,
                        confidence        = decision.confidence,
                        uncertain         = False,
                        impact            = "medium",
                        severity          = decision.severity,
                        timestamp         = datetime.utcnow().isoformat() + "Z",
                        raw_q_values      = None,
                    )

            else:
                # Z3 returned 'unknown' (timeout) — default to DENY
                act_approved = False
                approved = False
                violated.append("Z3_TIMEOUT")
                reasons.append(
                    f"Z3 solver timed out (500 ms) for action {action_name} "
                    "— defaulting to DENY"
                )
                self.logger.error(
                    "z3_solver_unknown",
                    action_name = action_name,
                    alert_id    = alert_id,
                    reason      = reasons[-1],
                )

            # ── C6: post-solver — always runs, never blocks ────────────────
            if is_credential_action:
                requires_audit = True
                self.logger.info(
                    "credential_action_audit_required",
                    constraint_name = C6,
                    action_name     = action_name,
                    alert_id        = alert_id,
                )

            # [IMMUNEX-PATCH] Bug 6: record network-blocking action ONLY if
            # approved per-action, using arrival_time not time.monotonic() here.
            if act_approved and is_network_block:
                self._record_action(source_ip, arrival_time)

        # Deduplicate violated constraint list while preserving order
        seen_v: set[str] = set()
        violated = [v for v in violated if not (v in seen_v or seen_v.add(v))]  # type: ignore

        reason_str = "; ".join(reasons) if reasons else "All constraints satisfied"

        verification = VerificationResult(
            approved                = approved,
            violated_constraints    = violated,
            substituted_action      = substituted,
            requires_audit_log      = requires_audit,
            requires_human_approval = requires_human,
            reason                  = reason_str,
        )

        self.logger.info(
            "verification_complete",
            alert_id          = alert_id,
            actions           = decision.actions,  # [IMMUNEX-PATCH] log full action list
            approved          = approved,
            violated          = violated,
            requires_audit    = requires_audit,
            requires_human    = requires_human,
            is_human_reviewed = is_human_reviewed,  # [IMMUNEX-PATCH] Bug 4: log the flag
        )

        return verification

    # ── Helpers ────────────────────────────────────────────────────────────

    def _is_trading_hours(self, timestamp: str) -> bool:
        """Returns True if *timestamp* falls within IST 09:00–17:00 Mon–Fri."""
        try:
            # FIX 2: handle both tz-aware and tz-naive strings.
            dt_parsed = datetime.fromisoformat(timestamp.rstrip("Z"))
            if dt_parsed.tzinfo is None:
                dt_parsed = pytz.utc.localize(dt_parsed)
            dt_ist   = dt_parsed.astimezone(_IST)
            in_week  = dt_ist.weekday() < 5                  # Mon–Fri
            mins     = dt_ist.hour * 60 + dt_ist.minute
            in_hours = (9 * 60) <= mins < (17 * 60)
            return in_week and in_hours
        except Exception as exc:
            self.logger.warning("trading_hours_parse_error", error=str(exc))
            return False   # fail open on parse error

    def _check_rate_limit(
        self,
        source_ip:    str,
        action_index: int,
        # [IMMUNEX-PATCH] Bug 6: callers supply the request arrival time so the
        # rate window is anchored to when the request hit the API, not when
        # this method executes (which could be much later due to queue waits).
        arrival_time: float | None = None,
    ) -> bool:
        """
        Returns True if source_ip has NOT yet hit the rate ceiling.
        Read-only — does not modify the rate window.
        Recording happens in _record_action(), called only from verify()
        after a successful per-action approval.

        [IMMUNEX-PATCH] Bug 6: uses arrival_time for the window boundary.
        """
        if action_index not in self._NETWORK_BLOCK_ACTIONS:
            return True

        now    = arrival_time if arrival_time is not None else time.monotonic()
        window = self._rate_window.get(source_ip, deque())
        cutoff = now - self._RATE_WINDOW_SECS

        # Count entries still within the sliding window
        recent = sum(1 for t in window if t >= cutoff)
        return recent < self._RATE_MAX_HITS

    def _record_action(
        self,
        source_ip:    str,
        # [IMMUNEX-PATCH] Bug 6: record the arrival_time, not time.monotonic() now
        arrival_time: float | None = None,
    ) -> None:
        """
        Appends the request arrival timestamp to the rate window for
        *source_ip* and evicts entries older than the window.
        Called exclusively from verify() after an approved network block.

        [IMMUNEX-PATCH] Bug 6: anchor timestamp = arrival_time, not
        time.monotonic() at execution time.
        """
        now    = arrival_time if arrival_time is not None else time.monotonic()
        window = self._rate_window.setdefault(source_ip, deque())
        window.append(now)
        cutoff = now - self._RATE_WINDOW_SECS
        while window and window[0] < cutoff:
            window.popleft()

    def _backup_exists(self) -> bool:
        """Returns True if at least one snapshot is listed in the registry."""
        path = self.backup_registry_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                registry = json.load(fh)
            return isinstance(registry, list) and len(registry) > 0
        except Exception as exc:
            self.logger.warning("backup_registry_read_error",
                                path=path, error=str(exc))
            return False


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _PASS = "\033[92m[PASS]\033[0m"
    _FAIL = "\033[91m[FAIL]\033[0m"

    def _make_decision(
        action_idx:        int,
        severity:          str  = "high",
        requires_approval: bool = False,
        timestamp:         str | None = None,
    ) -> ActionDecision:
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        return ActionDecision(
            alert_id          = str(uuid.uuid4()),
            action_index      = action_idx,
            action_name       = ACTION_NAMES[action_idx],
            actions           = [action_idx],
            action_names      = [ACTION_NAMES[action_idx]],
            action_categories = [get_action_category(action_idx)],
            requires_approval = requires_approval,
            confidence        = 0.9,
            uncertain         = False,
            impact            = "medium",
            severity          = severity,
            timestamp         = ts,
            raw_q_values      = None,
        )

    def _alert(source_ip: str = "10.0.0.1", dest_ip: str = "10.0.0.2") -> dict:
        return {
            "alert_id"       : str(uuid.uuid4()),
            "source_ip"      : source_ip,
            "destination_ip" : dest_ip,
        }

    results: list[tuple[str, bool, str]] = []

    # ── C1: quarantine_subnet (13) during IST trading hours, no human review ─
    # [IMMUNEX-PATCH] Bug 4 smoke test: is_human_reviewed=False must block
    trading_ts = "2026-03-24T06:00:00Z"  # 11:30 IST — inside trading window
    v1 = SafetyVerifier(mgmt_ip="192.168.1.1")

    d = _make_decision(13, severity="critical", requires_approval=False, timestamp=trading_ts)
    r = v1.verify(d, _alert(), is_human_reviewed=False)  # [IMMUNEX-PATCH] Bug 4
    ok = not r.approved and C1 in r.violated_constraints
    results.append(("C1: quarantine_subnet blocked during trading hours (no human review)", ok, r.reason))

    # ── C1b: same action WITH is_human_reviewed=True → must pass ──────────
    # [IMMUNEX-PATCH] Bug 4: override correctly unlocks C1 when human reviewed
    d2 = _make_decision(13, severity="critical", requires_approval=True, timestamp=trading_ts)
    r2 = v1.verify(d2, _alert(), is_human_reviewed=True)  # [IMMUNEX-PATCH] Bug 4
    results.append(("C1b: quarantine_subnet with is_human_reviewed=True passes", r2.approved, r2.reason))

    # ── C2: network action with low severity → denied ─────────────────────
    v2 = SafetyVerifier(mgmt_ip="192.168.1.1")
    d = _make_decision(10, severity="low")
    r = v2.verify(d, _alert())
    ok = not r.approved and C2 in r.violated_constraints
    results.append(("C2: block_source_ip blocked on low severity", ok, r.reason))

    # ── C3: block_source_ip with target_ip = management IP ───────────────
    # [IMMUNEX-PATCH] Bug 5: test using action_params["target_ip"]
    v3 = SafetyVerifier(mgmt_ip="10.1.1.1")
    d = _make_decision(10, severity="critical")
    # Pass target_ip explicitly (simulates firewall action payload)
    r = v3.verify(d, _alert(source_ip="10.0.0.5"), action_params={"target_ip": "10.1.1.1"})
    ok = not r.approved and C3 in r.violated_constraints
    results.append(("C3: block action targeting mgmt IP via action_params denied", ok, r.reason))

    # ── C4: rate limit — fixed arrival_time simulates real queue timing ───
    # [IMMUNEX-PATCH] Bug 6: supply a fixed arrival_time to avoid monotonic drift
    v4 = SafetyVerifier(mgmt_ip="127.0.0.1")
    _test_ip  = "203.0.113.42"
    _t0       = time.perf_counter()

    for i in range(5):
        d = _make_decision(10, severity="critical")
        rv = v4.verify(d, _alert(source_ip=_test_ip),
                       request_arrival_time=_t0 + i)   # [IMMUNEX-PATCH] Bug 6
        assert rv.approved, (
            f"C4 setup: iteration {i+1} unexpectedly denied — {rv.reason}"
        )

    d6 = _make_decision(10, severity="critical")
    r6 = v4.verify(d6, _alert(source_ip=_test_ip),
                   request_arrival_time=_t0 + 5)       # [IMMUNEX-PATCH] Bug 6
    ok = not r6.approved and C4 in r6.violated_constraints
    results.append(("C4: 6th network block on same IP denied (rate limit)", ok, r6.reason))

    # ── C5: rollback with no backup registry → substituted ────────────────
    v5 = SafetyVerifier(mgmt_ip="127.0.0.1", backup_registry_path="")
    d = _make_decision(33, severity="critical")   # rollback_filesystem_changes
    r = v5.verify(d, _alert())
    ok = (
        not r.approved
        and C5 in r.violated_constraints
        and r.substituted_action is not None
        and r.substituted_action.action_index == 47
    )
    results.append(("C5: rollback without backup → substituted action 47", ok, r.reason))

    # ── C6: credential action always approved but flags audit ──────────────
    v6 = SafetyVerifier(mgmt_ip="127.0.0.1")
    d = _make_decision(22, severity="high")   # disable_compromised_account
    r = v6.verify(d, _alert())
    ok = r.approved and r.requires_audit_log
    results.append(("C6: credential action (22) approved and requires audit", ok, r.reason))

    # ── Multi-action: one bad action in list blocks all ────────────────────
    # [IMMUNEX-PATCH] Multi-action smoke test
    v7 = SafetyVerifier(mgmt_ip="127.0.0.1")
    # actions=[13, 1]: action 13 during trading hours with no human review → blocked
    d_multi = ActionDecision(
        alert_id=str(uuid.uuid4()),
        action_index=13,
        action_name=ACTION_NAMES[13],
        actions=[13, 1],
        action_names=[ACTION_NAMES[13], ACTION_NAMES[1]],
        action_categories=[get_action_category(13), get_action_category(1)],
        requires_approval=False,
        confidence=0.9,
        uncertain=False,
        impact="high",
        severity="critical",
        timestamp=trading_ts,
        raw_q_values=None,
    )
    r_multi = v7.verify(d_multi, _alert(), is_human_reviewed=False)
    ok_multi = not r_multi.approved and C1 in r_multi.violated_constraints
    results.append(("Multi-action: action 13 in [13,1] blocks entire batch during trading hours", ok_multi, r_multi.reason))

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  IMMUNEX Layer 3 — SafetyVerifier Smoke Test [IMMUNEX-PATCH]")
    print("=" * 70)
    all_pass = True
    for desc, passed, reason in results:
        tag = _PASS if passed else _FAIL
        print(f"\n{tag}  {desc}")
        print(f"       reason : {reason}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("  \033[92mAll constraint checks passed.\033[0m")
    else:
        print("  \033[91mOne or more checks failed — see output above.\033[0m")
        sys.exit(1)
    print("=" * 70 + "\n")