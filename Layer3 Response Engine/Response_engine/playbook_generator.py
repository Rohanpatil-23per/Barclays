"""
IMMUNEX — Layer 3: Playbook Generator
=======================================
Calls a locally-hosted Llama-3 model (via Ollama) to produce a
human-readable SOC incident report and step-by-step response playbook.

Falls back to a rule-based template if Ollama is unavailable or times out,
ensuring the pipeline never stalls waiting for the LLM.

Dependency: pip install ollama
Runtime:    ollama pull llama3   (one-time download, ~4 GB)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
import ollama as ollama_client

from response_engine.response_engine import ActionDecision
from response_engine.safety_verifier import VerificationResult
from response_engine.action_executor import ExecutionResult
from response_engine.action_registry import ACTION_NAMES

# ── Logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# ── Risk level map (severity → risk label) ────────────────────────────────
_RISK_MAP: dict[str, str] = {
    "low":      "LOW",
    "medium":   "MEDIUM",
    "high":     "HIGH",
    "critical": "CRITICAL",
}

# ── Category playbook templates (fallback) ────────────────────────────────
_CATEGORY_PLAYBOOKS: dict[str, list[str]] = {
    "monitoring": [
        "1. Acknowledge alert in SIEM and assign to analyst.",
        "2. Correlate with recent events from same source IP.",
        "3. Increase log verbosity for affected host/subnet.",
        "4. Review baseline behaviour for anomaly confirmation.",
        "5. Escalate to Tier-2 if pattern persists for > 15 minutes.",
    ],
    "network": [
        "1. Confirm malicious IP/port via threat intelligence feeds.",
        "2. Apply firewall block rule to source IP.",
        "3. Update dynamic block list on NGFW (Palo Alto / Cisco).",
        "4. Validate block effectiveness via packet capture.",
        "5. Monitor for lateral movement from adjacent hosts.",
        "6. Notify NOC team via incident management platform.",
    ],
    "credential": [
        "1. Identify affected user account and timestamp of compromise.",
        "2. Disable/revoke active sessions and refresh tokens immediately.",
        "3. Force MFA re-authentication for the affected account.",
        "4. Reset credentials and notify user via out-of-band channel.",
        "5. Audit Active Directory logs for privilege escalation attempts.",
        "6. Review access logs for data exfiltration indicators.",
    ],
    "process": [
        "1. Capture memory dump of affected process before termination.",
        "2. Terminate or quarantine the malicious process/file.",
        "3. Submit sample to sandbox (Cuckoo/Joe Sandbox) for analysis.",
        "4. Run full EDR deep scan on affected host.",
        "5. Patch or update the exploited service.",
        "6. Restore from verified clean snapshot if integrity is compromised.",
    ],
    "data_protection": [
        "1. Identify data categories at risk (PII, PCI-DSS, banking data).",
        "2. Freeze or restrict access to affected data stores.",
        "3. Enable DLP policy to prevent further exfiltration.",
        "4. Notify Data Protection Officer (GDPR Article 33 — 72h window).",
        "5. Rotate encryption keys for compromised data stores.",
        "6. Commence forensic imaging for legal/regulatory evidence.",
    ],
}

_CATEGORY_COMPLIANCE: dict[str, str] = {
    "monitoring": (
        "Monitoring actions are low-impact. Ensure SIEM logs are retained per "
        "RBI cybersecurity circular (minimum 180 days). No GDPR notification required."
    ),
    "network": (
        "Network containment must be documented per DORA ICT incident reporting "
        "(Article 19). If external routing is disrupted, notify RBI within 6 hours. "
        "Ensure firewall rule changes are logged in the change management system."
    ),
    "credential": (
        "Credential actions affecting personal data require GDPR Article 32 technical "
        "safeguards. If breach notification threshold met, notify supervisory authority "
        "within 72 hours (GDPR Article 33). Log all AD changes for RBI audit trail."
    ),
    "process": (
        "Process containment on banking endpoints must follow RBI IT Framework "
        "change management procedures. Forensic integrity of memory dumps must be "
        "maintained for potential legal proceedings. DORA ICT testing requirements apply."
    ),
    "data_protection": (
        "Data protection actions trigger GDPR Article 33/34 assessment — evaluate "
        "whether personal data breach notification is required. RBI circular mandates "
        "immediate reporting of data exfiltration incidents. DORA Article 19 ICT "
        "incident classification must be completed within 4 hours."
    ),
}


# ── PlaybookReport ─────────────────────────────────────────────────────────

@dataclass
class PlaybookReport:
    alert_id:              str
    generated_at:          str          # ISO 8601
    incident_summary:      str
    threat_classification: str
    risk_level:            str
    action_taken:          str
    action_rationale:      str
    playbook_steps:        list[str]
    compliance_notes:      str
    recommended_followup:  list[str]
    raw_llm_response:      str
    generation_time_ms:    float


# ── PlaybookGenerator ──────────────────────────────────────────────────────

class PlaybookGenerator:
    """
    Generates SOC incident reports and response playbooks using Llama-3
    hosted locally via Ollama.

    Parameters
    ----------
    model       : Ollama model name, e.g. "llama3" or "llama3:8b".
    ollama_host : Base URL of the Ollama server (default: localhost:11434).
    timeout_s   : Hard timeout for Ollama inference (seconds).
    """

    _TIMEOUT_S: int = 30

    def __init__(
        self,
        model:       str = "llama3",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        self.model       = model
        self.ollama_host = ollama_host
        self.logger      = structlog.get_logger("playbook_generator")

        # Initialise the Ollama client
        self._client = ollama_client.Client(host=ollama_host)

        # Probe connectivity — warn but don't crash if unavailable
        self._ollama_available = self._probe_ollama()

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        alert:        dict,
        decision:     ActionDecision,
        verification: VerificationResult,
        execution:    ExecutionResult,
    ) -> PlaybookReport:
        """
        Generates a PlaybookReport for the given incident context.

        Falls back to a rule-based template if Ollama is unavailable,
        the call times out, or the LLM returns unparseable output.
        """
        t_start  = time.perf_counter()
        alert_id = decision.alert_id

        if not self._ollama_available:
            self.logger.warning(
                "ollama_unavailable_using_fallback",
                alert_id = alert_id,
                host     = self.ollama_host,
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            report  = self._fallback_report(alert, decision, execution)
            report.generation_time_ms = round(elapsed, 3)
            return report

        prompt   = self._build_prompt(alert, decision, verification, execution)
        raw_resp = self._call_ollama_with_timeout(prompt, alert_id)
        elapsed  = (time.perf_counter() - t_start) * 1000.0

        if raw_resp is None:
            # Timeout or exception during LLM call
            report = self._fallback_report(alert, decision, execution)
            report.generation_time_ms = round(elapsed, 3)
            return report

        parsed = self._parse_response(raw_resp, alert_id)

        return PlaybookReport(
            alert_id              = alert_id,
            generated_at          = datetime.utcnow().isoformat() + "Z",
            incident_summary      = parsed.get("incident_summary",
                                               self._default_summary(alert, decision)),
            threat_classification = parsed.get("threat_classification",
                                               alert.get("attack_type", "Unknown")),
            risk_level            = parsed.get("risk_level",
                                               _RISK_MAP.get(decision.severity, "UNKNOWN")),
            action_taken          = decision.action_name,
            action_rationale      = parsed.get("action_rationale",
                                               f"DQN selected {decision.action_name} "
                                               f"(confidence {decision.confidence:.2%})."),
            playbook_steps        = parsed.get("playbook_steps",
                                               _CATEGORY_PLAYBOOKS.get(decision.action_category, [])),
            compliance_notes      = parsed.get("compliance_notes",
                                               _CATEGORY_COMPLIANCE.get(decision.action_category, "")),
            recommended_followup  = parsed.get("recommended_followup", []),
            raw_llm_response      = raw_resp,
            generation_time_ms    = round(elapsed, 3),
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _probe_ollama(self) -> bool:
        """Returns True if the Ollama server responds to a list-models call."""
        try:
            self._client.list()
            self.logger.info("ollama_connectivity_ok", host=self.ollama_host, model=self.model)
            return True
        except Exception as exc:
            self.logger.warning(
                "ollama_connectivity_failed",
                host  = self.ollama_host,
                error = str(exc),
                hint  = "Install Ollama desktop app and run: ollama pull llama3",
            )
            return False

    def _call_ollama_with_timeout(
        self, prompt: str, alert_id: str
    ) -> str | None:
        """
        Calls Ollama in a thread-pool worker and enforces a hard timeout.
        Returns the raw string response, or None if timed out / failed.
        """
        def _call() -> str:
            response = self._client.chat(
                model    = self.model,
                messages = [
                    {
                        "role":    "system",
                        "content": (
                            "You are an expert cyber incident responder for a banking system. "
                            "You respond ONLY with valid JSON. No markdown. No explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"]

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                result = future.result(timeout=self._TIMEOUT_S)
                self.logger.info(
                    "llm_response_received",
                    alert_id       = alert_id,
                    response_chars = len(result),
                )
                return result
            except FuturesTimeoutError:
                self.logger.warning(
                    "ollama_timeout",
                    alert_id  = alert_id,
                    timeout_s = self._TIMEOUT_S,
                )
                return None
            except Exception as exc:
                self.logger.warning(
                    "ollama_call_failed",
                    alert_id = alert_id,
                    error    = str(exc),
                )
                return None

    def _build_prompt(
        self,
        alert:        dict,
        decision:     ActionDecision,
        verification: VerificationResult,
        execution:    ExecutionResult,
    ) -> str:
        """Builds the structured Llama-3 prompt."""
        constraints_str = (
            ", ".join(verification.violated_constraints)
            if verification.violated_constraints
            else "None"
        )
        # Concise action list for the model's context window
        action_list = "\n  ".join(
            f"{idx}: {name}" for idx, name in ACTION_NAMES.items()
        )

        return f"""INCIDENT CONTEXT:
  Alert ID:           {alert.get('alert_id', 'N/A')}
  Timestamp:          {alert.get('timestamp', 'N/A')}
  Source IP:          {alert.get('source_ip', 'N/A')}
  Destination IP:     {alert.get('destination_ip', 'N/A')}
  Protocol:           {alert.get('protocol', 'N/A')}
  Severity:           {alert.get('severity', 'N/A')}
  Attack Type:        {alert.get('attack_type', 'N/A')}
  Layer 2 Confidence: {alert.get('layer2_confidence', 0.0):.2%}

RESPONSE DECISION:
  Action Taken:       {decision.action_name} (index {decision.action_index})
  Category:           {decision.action_category}
  Execution Status:   {execution.status}
  Z3 Safety Verified: {verification.approved}
  Constraints Violated: {constraints_str}
  Requires Approval:  {decision.requires_approval}

ENVIRONMENT:
  Banking production system.
  Compliance requirements: RBI Cybersecurity Circular, GDPR Article 32,
  DORA ICT incident reporting (Article 19).

AVAILABLE ACTIONS (for context, 50 total):
  {action_list}

Respond with a JSON object containing EXACTLY these fields:
  "incident_summary"      : 2-3 sentence executive summary
  "threat_classification" : threat category/name
  "risk_level"            : one of LOW / MEDIUM / HIGH / CRITICAL
  "action_rationale"      : why this specific action was the correct response
  "playbook_steps"        : JSON array of 5-8 numbered step strings
  "compliance_notes"      : relevant RBI/GDPR/DORA obligations
  "recommended_followup"  : JSON array of 3-5 follow-up action strings

Output valid JSON only. No markdown fences. No preamble. No trailing text."""

    def _parse_response(self, raw: str, alert_id: str) -> dict[str, Any]:
        """
        Attempts to parse the LLM response as JSON.
        Strips markdown fences on first failure before retrying.
        Returns an empty dict on complete failure.
        """
        # Attempt 1: direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip ```json ... ``` fences
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE)
        try:
            return json.loads(stripped.strip())
        except json.JSONDecodeError:
            pass

        # Attempt 3: extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning(
            "llm_response_parse_failed",
            alert_id        = alert_id,
            raw_preview     = raw[:200],
        )
        return {}

    def _fallback_report(
        self,
        alert:    dict,
        decision: ActionDecision,
        execution: ExecutionResult,
    ) -> PlaybookReport:
        """
        Rule-based PlaybookReport used when Ollama is unavailable or fails.
        Guarantees the pipeline always returns a complete report object.
        """
        severity    = alert.get("severity", "unknown")
        attack_type = alert.get("attack_type", "Unknown Threat")
        category    = decision.action_category
        src_ip      = alert.get("source_ip", "unknown")

        return PlaybookReport(
            alert_id              = decision.alert_id,
            generated_at          = datetime.utcnow().isoformat() + "Z",
            incident_summary      = self._default_summary(alert, decision),
            threat_classification = attack_type,
            risk_level            = _RISK_MAP.get(severity, "UNKNOWN"),
            action_taken          = decision.action_name,
            action_rationale      = (
                f"IMMUNEX Dueling DQN selected '{decision.action_name}' "
                f"(action index {decision.action_index}) with confidence "
                f"{decision.confidence:.2%} based on the {attack_type} threat pattern "
                f"originating from {src_ip}. Z3 safety verification "
                f"{'approved' if execution.status != 'failed' else 'processed'} the action."
            ),
            playbook_steps        = _CATEGORY_PLAYBOOKS.get(category, [
                "1. Acknowledge alert and assign to on-call analyst.",
                "2. Confirm alert details with additional telemetry.",
                "3. Apply containment action as recommended by IMMUNEX.",
                "4. Document all actions in incident management system.",
                "5. Escalate if threat persists after initial containment.",
            ]),
            compliance_notes      = _CATEGORY_COMPLIANCE.get(
                category,
                "Ensure all incident response actions are logged per RBI cybersecurity "
                "framework and DORA ICT incident reporting obligations.",
            ),
            recommended_followup  = [
                f"Conduct threat hunt across network segment containing {src_ip}.",
                "Update threat intelligence feeds with IoCs from this incident.",
                "Review and tighten firewall/ACL rules to prevent recurrence.",
                "Complete post-incident review within 5 business days (DORA requirement).",
                "Verify backup integrity for all affected systems.",
            ],
            raw_llm_response      = "[FALLBACK — Ollama unavailable or timed out]",
            generation_time_ms    = 0.0,
        )

    @staticmethod
    def _default_summary(alert: dict, decision: ActionDecision) -> str:
        return (
            f"A {alert.get('severity', 'unknown').upper()} severity "
            f"{alert.get('attack_type', 'threat')} was detected from "
            f"{alert.get('source_ip', 'unknown')} targeting "
            f"{alert.get('destination_ip', 'unknown')} via "
            f"{alert.get('protocol', 'unknown')}. "
            f"IMMUNEX Layer 3 autonomously selected '{decision.action_name}' "
            f"as the optimal containment action. "
            f"The response pipeline completed with status: {decision.action_category}."
        )


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dataclasses import asdict
    from response_engine.action_registry import get_action_category

    _PASS = "\033[92m[PASS]\033[0m"
    _FAIL = "\033[91m[FAIL]\033[0m"

    # ── Synthetic incident context ────────────────────────────────────────
    _alert_id = str(uuid.uuid4())
    _alert = {
        "alert_id"         : _alert_id,
        "timestamp"        : datetime.utcnow().isoformat() + "Z",
        "source_ip"        : "203.0.113.42",
        "destination_ip"   : "10.0.1.100",
        "source_port"      : 4444,
        "destination_port" : 443,
        "protocol"         : "TCP",
        "severity"         : "critical",
        "attack_type"      : "C2_Beacon",
        "layer2_confidence": 0.97,
    }

    _decision = ActionDecision(
        alert_id          = _alert_id,
        action_index      = 19,
        action_name       = "block_c2_domain",
        action_category   = "network",
        requires_approval = False,
        confidence        = 0.97,
        severity          = "critical",
        timestamp         = _alert["timestamp"],
        raw_q_values      = None,
    )

    from response_engine.safety_verifier import VerificationResult as VR
    _verification = VR(
        approved                = True,
        violated_constraints    = [],
        substituted_action      = None,
        requires_audit_log      = False,
        requires_human_approval = False,
        reason                  = "All constraints satisfied",
    )

    _execution = ExecutionResult(
        alert_id          = _alert_id,
        action_index      = 19,
        action_name       = "block_c2_domain",
        status            = "simulated",
        dry_run           = True,
        execution_time_ms = 0.42,
        output            = {"action": "dns_block", "system": "Cisco_Umbrella"},
        error             = None,
    )

    results: list[tuple[str, bool, str]] = []

    # ── Test 1: Forced fallback (bad host) ────────────────────────────────
    print("\nTest 1: Forced fallback (bad Ollama host)...")
    gen_fallback = PlaybookGenerator(
        model       = "llama3",
        ollama_host = "http://localhost:19999",  # non-existent port
    )
    t0  = time.perf_counter()
    rep = gen_fallback.generate(_alert, _decision, _verification, _execution)
    elapsed = (time.perf_counter() - t0) * 1000.0

    ok = (
        isinstance(rep, PlaybookReport)
        and rep.alert_id == _alert_id
        and len(rep.playbook_steps) >= 5
        and len(rep.recommended_followup) >= 3
        and rep.raw_llm_response == "[FALLBACK — Ollama unavailable or timed out]"
    )
    results.append(("Fallback path: valid PlaybookReport returned", ok,
                    f"steps={len(rep.playbook_steps)}, "
                    f"followup={len(rep.recommended_followup)}, "
                    f"t={elapsed:.1f}ms"))

    # ── Test 2: Live Ollama (if available) ───────────────────────────────
    print("\nTest 2: Live Ollama path (http://localhost:11434)...")
    gen_live = PlaybookGenerator(model="llama3")
    if gen_live._ollama_available:
        rep_live = gen_live.generate(_alert, _decision, _verification, _execution)
        ok_live = (
            isinstance(rep_live, PlaybookReport)
            and rep_live.alert_id == _alert_id
            and rep_live.raw_llm_response != "[FALLBACK — Ollama unavailable or timed out]"
            and len(rep_live.playbook_steps) > 0
        )
        results.append(("Live path: Ollama responded and report parsed",
                         ok_live,
                         f"risk={rep_live.risk_level}, "
                         f"t={rep_live.generation_time_ms:.0f}ms"))
    else:
        results.append(("Live path: Ollama not running — skipped (expected in CI)",
                         True,
                         "Install Ollama desktop app and run: ollama pull llama3"))

    # ── Test 3: JSON parser — with markdown fences ────────────────────────
    gen_parse = PlaybookGenerator(ollama_host="http://localhost:19999")
    fenced_json = (
        "```json\n"
        '{"incident_summary": "Test", "threat_classification": "SQLi", '
        '"risk_level": "HIGH", "action_rationale": "test", '
        '"playbook_steps": ["Step 1"], "compliance_notes": "RBI", '
        '"recommended_followup": ["Action 1"]}\n'
        "```"
    )
    parsed = gen_parse._parse_response(fenced_json, "test-id")
    ok_parse = parsed.get("risk_level") == "HIGH" and "playbook_steps" in parsed
    results.append(("JSON parser: markdown-fenced response parsed correctly",
                     ok_parse, str(parsed.get("risk_level"))))

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  IMMUNEX Layer 3 — PlaybookGenerator Smoke Test")
    print("=" * 70)
    all_pass = True
    for desc, passed, detail in results:
        tag = _PASS if passed else _FAIL
        print(f"\n{tag}  {desc}")
        print(f"       detail : {detail}")
        if not passed:
            all_pass = False

    # Print a sample report section
    print("\n--- Sample PlaybookReport (fallback) ---")
    sample = {k: v for k, v in asdict(rep).items() if k != "raw_llm_response"}
    print(json.dumps(sample, indent=2))

    print("\n" + "=" * 70)
    if all_pass:
        print("  \033[92mAll PlaybookGenerator checks passed.\033[0m")
    else:
        print("  \033[91mSome checks failed — see output above.\033[0m")
        sys.exit(1)
    print("=" * 70 + "\n")
