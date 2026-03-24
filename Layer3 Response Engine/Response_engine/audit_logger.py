"""
IMMUNEX — Layer 3: Audit Logger
=================================
Immutable JSON Lines compliance audit trail.
Every incident decision, approval request, and approval outcome is
written to a daily .jsonl file using atomic os.replace() writes.

File format : audit_logs/audit_YYYY-MM-DD.jsonl
Line format : one JSON object per line (JSON Lines / NDJSON)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from response_engine.response_engine import ActionDecision
from response_engine.safety_verifier import VerificationResult
from response_engine.action_executor import ExecutionResult
from response_engine.playbook_generator import PlaybookReport

# Version stamped on every log entry
_IMMUNEX_VERSION = "3.0.0"
_LAYER           = "3-immune-response"


class AuditLogger:
    """
    Immutable JSON Lines audit logger for RBI / GDPR / DORA compliance.

    All writes are atomic — each append writes to a .tmp file in the same
    directory then calls os.replace() to swap it in, preventing corruption
    on crash or power loss.

    Parameters
    ----------
    log_dir : str   Directory where daily .jsonl files are written.
                    Created automatically if it does not exist.
    """

    def __init__(self, log_dir: str = "audit_logs") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    # ── Path helpers ───────────────────────────────────────────────────────

    def _daily_path(self) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{date_str}.jsonl")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Atomic write ───────────────────────────────────────────────────────

    def _atomic_append(self, entry: dict) -> None:
        """
        Appends *entry* as a JSON line to today's log file atomically.
        Writes to a .tmp sibling file, then os.replace() swaps it in.
        On any failure the .tmp file is removed and the exception re-raised.
        """
        filepath = self._daily_path()
        line     = json.dumps(entry, default=str) + "\n"

        fd, tmp_path = tempfile.mkstemp(
            dir    = self.log_dir,
            prefix = "audit_",
            suffix = ".tmp",
            text   = True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                # Copy existing content first, then append the new line
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as orig_fh:
                        shutil.copyfileobj(orig_fh, tmp_fh)
                tmp_fh.write(line)
            os.replace(tmp_path, filepath)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ── Dataclass / model serialiser ──────────────────────────────────────

    @staticmethod
    def _to_dict(obj: Any) -> dict | None:
        """Converts dataclasses, Pydantic models, dicts, or None to dict."""
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        return None

    # ── Public API ─────────────────────────────────────────────────────────

    def log_incident(
        self,
        alert:        dict,
        decision:     ActionDecision,
        verification: VerificationResult,
        execution:    ExecutionResult | None,
        playbook:     PlaybookReport  | None,
    ) -> str:
        """
        Logs a complete incident response decision.

        Returns the audit_entry_id (UUID string) in all cases —
        including on write failure, so the caller always gets a
        consistent identifier regardless of I/O errors.
        """
        audit_entry_id = str(uuid.uuid4())

        entry = {
            "audit_entry_id":  audit_entry_id,
            "logged_at":       self._utc_now(),
            "immunex_version": _IMMUNEX_VERSION,
            "layer":           _LAYER,
            "event_type":      "incident_decision",
            "alert":           alert,
            "decision":        self._to_dict(decision),
            "verification":    self._to_dict(verification),
            "execution":       self._to_dict(execution),
            "playbook":        self._to_dict(playbook),
        }

        try:
            self._atomic_append(entry)
        except Exception as exc:
            # FIX: return the SAME audit_entry_id on error so the caller
            # can still correlate — the original code generated a second UUID.
            print(f"[AuditLogger] FAILED to write incident log: {exc}",
                  file=sys.stderr)

        return audit_entry_id

    def log_approval_request(
        self,
        alert_id:     str,
        action_name:  str,
        requested_at: str,
        requested_by: str = "system",
    ) -> str:
        """Logs a high-impact action queued for human approval."""
        audit_entry_id = str(uuid.uuid4())

        entry = {
            "audit_entry_id":  audit_entry_id,
            "logged_at":       self._utc_now(),
            "immunex_version": _IMMUNEX_VERSION,
            "layer":           _LAYER,
            "event_type":      "approval_request",
            "alert_id":        alert_id,
            "action_name":     action_name,
            "requested_at":    requested_at,
            "requested_by":    requested_by,
        }

        try:
            self._atomic_append(entry)
        except Exception as exc:
            print(f"[AuditLogger] FAILED to write approval request log: {exc}",
                  file=sys.stderr)

        return audit_entry_id

    def log_approval_decision(
        self,
        alert_id:  str,
        approved:  bool,
        approver:  str,
        reason:    str,
    ) -> str:
        """Logs the outcome of a human approval decision."""
        audit_entry_id = str(uuid.uuid4())

        entry = {
            "audit_entry_id":  audit_entry_id,
            "logged_at":       self._utc_now(),
            "immunex_version": _IMMUNEX_VERSION,
            "layer":           _LAYER,
            "event_type":      "approval_decision",
            "alert_id":        alert_id,
            "approved":        approved,
            "approver":        approver,
            "reason":          reason,
        }

        try:
            self._atomic_append(entry)
        except Exception as exc:
            print(f"[AuditLogger] FAILED to write approval decision log: {exc}",
                  file=sys.stderr)

        return audit_entry_id

    def query_by_alert_id(self, alert_id: str) -> list[dict]:
        """
        Returns all log entries for *alert_id* from today's log file.

        Checks both top-level `alert_id` (approval events) and
        nested `alert.alert_id` (incident_decision events).
        """
        filepath = self._daily_path()
        results: list[dict] = []

        if not os.path.exists(filepath):
            return results

        with open(filepath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Approval events store alert_id at top level
                entry_aid = entry.get("alert_id")
                # Incident events store it nested under "alert"
                if entry_aid is None and isinstance(entry.get("alert"), dict):
                    entry_aid = entry["alert"].get("alert_id")

                if entry_aid == alert_id:
                    results.append(entry)

        return results


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from response_engine.action_registry import get_action_category

    print("=" * 55)
    print("  IMMUNEX Layer 3 — AuditLogger Smoke Test")
    print("=" * 55)

    _log_dir = "tmp_audit_logs_smoke"
    logger   = AuditLogger(log_dir=_log_dir)

    _alert_id = "TEST-" + str(uuid.uuid4())[:8].upper()

    # Build synthetic objects using the REAL field names from each dataclass
    # FIX: smoke test previously used non-existent fields (suggested_playbook,
    # constraints_checked, violations) which crashed on construction.
    _alert = {
        "alert_id":          _alert_id,
        "severity":          "high",
        "source_ip":         "10.0.0.42",
        "destination_ip":    "10.0.0.1",
        "attack_type":       "SQLi",
        "layer2_confidence": 0.91,
    }

    _decision = ActionDecision(
        alert_id          = _alert_id,
        action_index      = 10,
        action_name       = "block_source_ip",
        action_category   = get_action_category(10),
        requires_approval = False,
        confidence        = 0.91,
        severity          = "high",
        timestamp         = datetime.now(timezone.utc).isoformat(),
        raw_q_values      = None,
    )

    _verification = VerificationResult(
        approved                = True,
        violated_constraints    = [],
        substituted_action      = None,
        requires_audit_log      = False,
        requires_human_approval = False,
        reason                  = "All constraints satisfied",
    )

    _execution = ExecutionResult(
        alert_id          = _alert_id,
        action_index      = 10,
        action_name       = "block_source_ip",
        status            = "simulated",
        dry_run           = True,
        execution_time_ms = 0.5,
        output            = {"target_ip": "10.0.0.42", "rule": "DROP"},
        error             = None,
    )

    _playbook = PlaybookReport(
        alert_id              = _alert_id,
        generated_at          = datetime.now(timezone.utc).isoformat(),
        incident_summary      = "SQLi attack detected and blocked.",
        threat_classification = "SQLi",
        risk_level            = "HIGH",
        action_taken          = "block_source_ip",
        action_rationale      = "Source IP blocked to prevent further injection.",
        playbook_steps        = ["Step 1", "Step 2", "Step 3"],
        compliance_notes      = "RBI circular applies.",
        recommended_followup  = ["Review DB logs", "Patch application"],
        raw_llm_response      = "[FALLBACK]",
        generation_time_ms    = 1.0,
    )

    _PASS = "\033[92m[PASS]\033[0m"
    _FAIL = "\033[91m[FAIL]\033[0m"
    results: list[tuple[str, bool]] = []

    # 1. Log incident
    eid1 = logger.log_incident(_alert, _decision, _verification, _execution, _playbook)
    ok1  = isinstance(eid1, str) and len(eid1) == 36
    results.append(("log_incident() returns valid UUID", ok1))

    # 2. Log approval request
    eid2 = logger.log_approval_request(
        _alert_id, "block_source_ip", datetime.now(timezone.utc).isoformat()
    )
    ok2 = isinstance(eid2, str) and len(eid2) == 36
    results.append(("log_approval_request() returns valid UUID", ok2))

    # 3. Log approval decision
    eid3 = logger.log_approval_decision(
        _alert_id, approved=True, approver="analyst_01", reason="Confirmed malicious"
    )
    ok3 = isinstance(eid3, str) and len(eid3) == 36
    results.append(("log_approval_decision() returns valid UUID", ok3))

    # 4. Query by alert_id — should find all 3 entries
    entries = logger.query_by_alert_id(_alert_id)
    ok4 = len(entries) == 3
    results.append((f"query_by_alert_id() returns 3 entries (got {len(entries)})", ok4))

    # 5. Log file is valid JSON Lines
    from pathlib import Path
    log_file = Path(_log_dir) / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    lines    = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    ok5      = all("audit_entry_id" in l and "immunex_version" in l for l in lines)
    results.append(("Log file is valid JSON Lines with required fields", ok5))

    print()
    all_pass = True
    for desc, passed in results:
        tag = _PASS if passed else _FAIL
        print(f"{tag}  {desc}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("\033[92m  All 5 checks passed.\033[0m")
    else:
        print("\033[91m  Some checks failed.\033[0m")

    # Cleanup
    shutil.rmtree(_log_dir, ignore_errors=True)
    print("=" * 55 + "\n")