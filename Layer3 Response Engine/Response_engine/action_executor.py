"""
IMMUNEX — Layer 3: Action Executor
====================================
Dispatches verified ActionDecisions to their stub handlers.
Each stub logs what it *would* do in production and returns structured output.
Real API calls can be wired in by replacing stub bodies.

Handler map (50 actions)
------------------------
  Monitoring      (0–9)   : do_nothing → generate_soc_report
  Network         (10–19) : block_source_ip → block_c2_domain
  Credential      (20–29) : revoke_user_session → audit_active_sessions
  Process         (30–39) : kill_malicious_process → sandbox_suspicious_binary
  Data Protection (40–49) : encrypt_sensitive_data_at_rest → activate_honeypot
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import structlog

from response_engine_module import ActionDecision

# ── Logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# ── Validation ─────────────────────────────────────────────────────────────

def validate_action_params(params: dict) -> bool:
    """
    Validates parameter types, bounds, and safety before execution.
    Specifically checks 'duration' fields for >= 0, and scans string 
    inputs for dangerous shell injection characters.
    """
    unsafe_chars = {';', '&', '|', '`', '$', '>'}
    
    for key, value in params.items():
        if "duration" in key or "timeout" in key:
            try:
                val = int(value)
                if val < 0:
                    return False
            except (ValueError, TypeError):
                if isinstance(value, str) and value.lower() in ("indefinite", "forever"):
                    pass
                else:
                    return False
                    
        if isinstance(value, str):
            if any(c in value for c in unsafe_chars):
                return False
                
    return True


# ── ExecutionResult ────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    alert_id:          str
    action_index:      int | None
    action_name:       str | None
    actions:           list[int]
    action_names:      list[str]
    status:            str          # "executed" | "simulated" | "failed" | "pending_approval" | "partial_success"
    dry_run:           bool
    execution_time_ms: float
    output:            dict
    validation_status: dict
    error:             str | None = None


# ── ActionExecutor ─────────────────────────────────────────────────────────

class ActionExecutor:
    """
    Dispatches an ActionDecision to the appropriate handler stub.

    Parameters
    ----------
    dry_run : bool
        True  → status="simulated", no real system calls attempted.
        False → status="executed",  handler bodies would perform real calls.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.logger  = structlog.get_logger("action_executor")

        # Build dispatch table: action_index → handler method
        self._dispatch: dict[int, Callable[[dict], dict]] = {
            0:  self._handle_do_nothing,
            1:  self._handle_increase_log_verbosity,
            2:  self._handle_trigger_soc_alert,
            3:  self._handle_snapshot_memory,
            4:  self._handle_capture_network_traffic,
            5:  self._handle_enable_deep_packet_inspection,
            6:  self._handle_flag_for_human_review,
            7:  self._handle_escalate_to_tier2,
            8:  self._handle_notify_incident_manager,
            9:  self._handle_generate_soc_report,
            10: self._handle_block_source_ip,
            11: self._handle_block_destination_ip,
            12: self._handle_isolate_endpoint,
            13: self._handle_quarantine_subnet,
            14: self._handle_disable_external_routing,
            15: self._handle_block_suspicious_port,
            16: self._handle_rate_limit_source,
            17: self._handle_drop_malicious_packets,
            18: self._handle_null_route_attacker,
            19: self._handle_block_c2_domain,
            20: self._handle_revoke_user_session,
            21: self._handle_force_mfa_reauthentication,
            22: self._handle_disable_compromised_account,
            23: self._handle_reset_service_account_password,
            24: self._handle_revoke_api_key,
            25: self._handle_restrict_admin_privileges,
            26: self._handle_enforce_least_privilege,
            27: self._handle_disable_lateral_movement_path,
            28: self._handle_lock_privileged_account,
            29: self._handle_audit_active_sessions,
            30: self._handle_kill_malicious_process,
            31: self._handle_suspend_suspicious_process,
            32: self._handle_quarantine_malicious_file,
            33: self._handle_rollback_filesystem_changes,
            34: self._handle_restore_from_clean_snapshot,
            35: self._handle_run_edr_deep_scan,
            36: self._handle_patch_vulnerable_service,
            37: self._handle_disable_autorun_mechanisms,
            38: self._handle_terminate_remote_desktop_session,
            39: self._handle_sandbox_suspicious_binary,
            40: self._handle_encrypt_sensitive_data_at_rest,
            41: self._handle_revoke_data_export_permissions,
            42: self._handle_block_usb_exfiltration,
            43: self._handle_disable_email_forwarding_rules,
            44: self._handle_watermark_sensitive_documents,
            45: self._handle_enable_dlp_policy,
            46: self._handle_freeze_database_writes,
            47: self._handle_backup_critical_data,
            48: self._handle_rotate_encryption_keys,
            49: self._handle_activate_honeypot,
        }

        self.logger.info("action_executor_initialised", dry_run=dry_run)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _extract_params_for_action(self, action_index: int, alert: dict) -> dict:
        """Extract parameters per action (handler-specific or derived)."""
        excluded = {"alert_id", "timestamp", "severity", "attack_type", "layer2_confidence", "feature_vector"}
        return {k: v for k, v in alert.items() if k not in excluded}

    # ── Public API ─────────────────────────────────────────────────────────

    def execute(
        self,
        decision:         ActionDecision,
        alert:            dict,
        approval_granted: bool = False,
    ) -> ExecutionResult:
        """
        Executes (or simulates) the action described by *decision*.

        Parameters
        ----------
        decision         : ActionDecision from ResponseEngine
        alert            : Original Layer 2 alert dict
        approval_granted : Must be True for high-impact actions; otherwise
                           returns status="pending_approval" immediately.
        """
        alert_id    = decision.alert_id
        action_idx  = decision.action_index
        action_name = decision.action_name
        actions     = decision.actions
        action_names= decision.action_names

        # ── Approval gate ──────────────────────────────────────────────────
        if decision.requires_approval and not approval_granted:
            self.logger.warning(
                "actions_pending_approval",
                alert_id    = alert_id,
                actions     = action_names,
            )
            return ExecutionResult(
                alert_id          = alert_id,
                action_index      = action_idx,
                action_name       = action_name,
                actions           = actions,
                action_names      = action_names,
                status            = "pending_approval",
                dry_run           = self.dry_run,
                execution_time_ms = 0.0,
                output            = {"message": "Awaiting human approval before execution"},
                validation_status = {"pre_check": "skipped", "post_check": "skipped"},
                error             = None,
            )

        t_start = time.perf_counter()

        # ── Pre-execution validation layer ─────────────────────────────────
        # Basic check to ensure a target entity exists in the alert before proceeding
        pre_check_passed = bool(alert.get("source_ip") or alert.get("destination_ip") or alert.get("user") or alert.get("process_name"))
        validation_status = {"pre_check": "passed" if pre_check_passed else "failed"}

        if not pre_check_passed:
            self.logger.error("pre_execution_validation_failed", alert_id=alert_id, reason="No target entity found")
            return ExecutionResult(
                alert_id          = alert_id,
                action_index      = action_idx,
                action_name       = action_name,
                actions           = actions,
                action_names      = action_names,
                status            = "failed",
                dry_run           = self.dry_run,
                execution_time_ms = 0.0,
                output            = {},
                validation_status = validation_status,
                error             = "Pre-execution validation failed: No target entity found in alert.",
            )

        # ── Dispatching ────────────────────────────────────────────────────
        outputs: dict = {}
        errors: list[str] = []
        status: str = "simulated" if self.dry_run else "executed"

        for idx, name in zip(actions, action_names):
            # Parameter validation
            params = self._extract_params_for_action(idx, alert)
            if not validate_action_params(params):
                self.logger.warning("action_validation_failed", action_index=idx, action_name=name, reason="unsafe parameters extracted")
                errors.append(f"{name} skipped: parameter validation failed")
                continue

            handler = self._dispatch.get(idx)
            if handler is None:
                err_msg = f"No handler registered for action_index={idx}"
                errors.append(err_msg)
                self.logger.error("handler_not_found", alert_id=alert_id, action_index=idx)
                continue

            try:
                outputs[name] = handler(alert)
            except Exception as exc:
                errors.append(f"{name} failed: {exc}")
                self.logger.error(
                    "handler_exception",
                    alert_id    = alert_id,
                    action_name = name,
                    error       = str(exc),
                )

        if len(errors) == len(actions):
            status = "failed"
        elif errors:
            status = "partial_success"

        # ── Post-execution validation layer ────────────────────────────────
        # Here we mock checking if the system state actually reflects the change
        validation_status["post_check"] = "passed" if status in ("executed", "simulated", "partial_success") else "failed"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        final_error = " | ".join(errors) if errors else None

        self.logger.info(
            "actions_executed",
            alert_id          = alert_id,
            actions           = action_names,
            status            = status,
            dry_run           = self.dry_run,
            execution_time_ms = round(elapsed_ms, 3),
        )

        return ExecutionResult(
            alert_id          = alert_id,
            action_index      = action_idx,
            action_name       = action_name,
            actions           = actions,
            action_names      = action_names,
            status            = status,
            dry_run           = self.dry_run,
            execution_time_ms = round(elapsed_ms, 3),
            output            = outputs,
            validation_status = validation_status,
            error             = final_error,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MONITORING HANDLERS  (actions 0–9)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_do_nothing(self, alert: dict) -> dict:
        self.logger.info("[STUB] do_nothing — no system interaction required",
                         alert_id=alert.get("alert_id"))
        return {"action": "no_op"}

    def _handle_increase_log_verbosity(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            "[STUB] Would call SIEM API (Splunk/QRadar): set log level=DEBUG "
            f"for source {src}"
        )
        return {"target": src, "log_level": "DEBUG", "system": "SIEM/Splunk"}

    def _handle_trigger_soc_alert(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would POST to SOC ticketing system (ServiceNow/JIRA): "
            f"open P1 ticket for alert {alert.get('alert_id')}"
        )
        return {
            "ticket_system": "ServiceNow",
            "priority":      "P1",
            "alert_id":      alert.get("alert_id"),
            "queued":        True,
        }

    def _handle_snapshot_memory(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR API (CrowdStrike): trigger memory dump on host {src}"
        )
        return {"host": src, "dump_type": "full_memory", "edr": "CrowdStrike"}

    def _handle_capture_network_traffic(self, alert: dict) -> dict:
        src  = alert.get("source_ip", "unknown")
        dest = alert.get("destination_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call TAP/mirror API: start pcap capture "
            f"for flow {src} → {dest}"
        )
        return {"src": src, "dest": dest, "capture": "pcap", "duration_s": 300}

    def _handle_enable_deep_packet_inspection(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call NGFW API (Palo Alto): enable DPI policy "
            f"for source subnet of {src}"
        )
        return {"firewall": "PaloAlto_NGFW", "policy": "DPI_enabled", "target": src}

    def _handle_flag_for_human_review(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would update SIEM case (QRadar): mark alert "
            f"{alert.get('alert_id')} for analyst review"
        )
        return {"alert_id": alert.get("alert_id"), "flag": "human_review", "system": "QRadar"}

    def _handle_escalate_to_tier2(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call paging system (PagerDuty): page Tier-2 analyst "
            f"for alert {alert.get('alert_id')}"
        )
        return {"paged": True, "tier": 2, "system": "PagerDuty", "alert_id": alert.get("alert_id")}

    def _handle_notify_incident_manager(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would send SMS/email via notification gateway: "
            f"alert IM on alert {alert.get('alert_id')}"
        )
        return {"notified": True, "channel": "SMS+email", "alert_id": alert.get("alert_id")}

    def _handle_generate_soc_report(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would trigger SOC report pipeline (internal IMMUNEX): "
            f"generate PDF/HTML report for alert {alert.get('alert_id')}"
        )
        return {"report_queued": True, "format": "PDF+HTML", "alert_id": alert.get("alert_id")}

    # ══════════════════════════════════════════════════════════════════════
    # NETWORK CONTAINMENT HANDLERS  (actions 10–19)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_block_source_ip(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        duration = alert.get("block_duration", "indefinite")  # Parameterized action support
        self.logger.info(
            f"[STUB] Would call firewall API: iptables -A INPUT -s {src} -j DROP "
            f"for duration {duration}"
        )
        return {"target_ip": src, "rule": "DROP", "duration": duration, "firewall": "iptables/PaloAlto"}

    def _handle_block_destination_ip(self, alert: dict) -> dict:
        dest = alert.get("destination_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call firewall API: iptables -A OUTPUT -d {dest} -j DROP"
        )
        return {"target_ip": dest, "rule": "DROP", "direction": "outbound", "firewall": "iptables"}

    def _handle_isolate_endpoint(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR API (CrowdStrike/SentinelOne): contain host {host} "
            "— cuts all network except EDR management channel"
        )
        return {"host": host, "edr_action": "contain", "vlan_isolation": True, "edr": "CrowdStrike"}

    def _handle_quarantine_subnet(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call SDN controller (Cisco ACI/VMware NSX): "
            f"apply quarantine micro-segment policy to subnet of {src}"
        )
        return {"subnet": src + "/24", "policy": "quarantine", "sdn": "CiscoACI/NSX"}

    def _handle_disable_external_routing(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call BGP/router API: withdraw external routes "
            "— shuts off internet egress for affected segment"
        )
        return {"action": "bgp_withdraw", "scope": "external_routes", "router": "Cisco_BGP"}

    def _handle_block_suspicious_port(self, alert: dict) -> dict:
        port  = alert.get("destination_port", 0)
        proto = alert.get("protocol", "TCP")
        self.logger.info(
            f"[STUB] Would call firewall API: iptables -A INPUT -p {proto} --dport {port} -j DROP"
        )
        return {"port": port, "protocol": proto, "rule": "DROP", "firewall": "iptables"}

    def _handle_rate_limit_source(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call NGFW/WAF API: apply rate limit 10 req/s for source {src}"
        )
        return {"target": src, "rate_limit": "10req/s", "system": "PaloAlto_WAF"}

    def _handle_drop_malicious_packets(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call IDPS API (Snort/Suricata): add drop rule for "
            f"packets from {src} matching attack signature"
        )
        return {"target": src, "action": "drop", "idps": "Snort/Suricata"}

    def _handle_null_route_attacker(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call router API: ip route {src}/32 Null0 — RTBH null-routing"
        )
        return {"target": src, "route": "Null0", "technique": "RTBH", "router": "Cisco"}

    def _handle_block_c2_domain(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call DNS/proxy API (Palo Alto DNS Security/Umbrella): "
            "add C2 domain to block list"
        )
        return {"action": "dns_block", "system": "Cisco_Umbrella/PaloAlto_DNS", "category": "C2"}

    # ══════════════════════════════════════════════════════════════════════
    # CREDENTIAL & ACCESS HANDLERS  (actions 20–29)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_revoke_user_session(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call AD/LDAP API: disable active session tokens "
            f"for user authenticated from {src}"
        )
        return {"action": "session_revoke", "target": src, "system": "ActiveDirectory/LDAP"}

    def _handle_force_mfa_reauthentication(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call MFA provider API (Duo/Microsoft Entra): "
            f"invalidate existing MFA tokens for user on {src}"
        )
        return {"action": "mfa_force_reauth", "target": src, "mfa_provider": "Duo/Entra"}

    def _handle_disable_compromised_account(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call AD/LDAP: userAccountControl=ACCOUNTDISABLE "
            f"for account associated with {src}"
        )
        return {"action": "account_disable", "target": src, "system": "ActiveDirectory"}

    def _handle_reset_service_account_password(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call AD API / CyberArk PAM: rotate password for "
            "service account associated with alerting host"
        )
        return {"action": "password_reset", "account_type": "service", "pam": "CyberArk"}

    def _handle_revoke_api_key(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call API Gateway (Kong/Apigee): invalidate API key "
            "associated with source of suspicious traffic"
        )
        return {"action": "api_key_revoke", "gateway": "Kong/Apigee"}

    def _handle_restrict_admin_privileges(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call AD / PAM (CyberArk): remove admin group membership "
            f"for user on {src} — enforce least privilege"
        )
        return {"action": "admin_restrict", "target": src, "pam": "CyberArk"}

    def _handle_enforce_least_privilege(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call PAM / IAM policy engine: apply least-privilege "
            "RBAC ruleset to alerting user account"
        )
        return {"action": "least_privilege_enforce", "system": "CyberArk/IAM"}

    def _handle_disable_lateral_movement_path(self, alert: dict) -> dict:
        src  = alert.get("source_ip", "unknown")
        dest = alert.get("destination_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call SDN/ACL API: deny east-west traffic "
            f"{src} → {dest} on VLAN"
        )
        return {"src": src, "dest": dest, "action": "acl_deny", "system": "SDN/VLAN"}

    def _handle_lock_privileged_account(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call CyberArk/AD: lock privileged (admin/service) "
            f"account associated with {src}"
        )
        return {"action": "privileged_account_lock", "target": src, "pam": "CyberArk"}

    def _handle_audit_active_sessions(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call SIEM / AD: enumerate and log all active sessions "
            "— output to audit trail (WORM storage)"
        )
        return {"action": "session_audit", "storage": "WORM", "system": "QRadar/AD"}

    # ══════════════════════════════════════════════════════════════════════
    # PROCESS & ENDPOINT HANDLERS  (actions 30–39)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_kill_malicious_process(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR API (CrowdStrike/SentinelOne): terminate PID "
            f"matching malicious hash / behaviour signature on {host}"
        )
        return {"host": host, "action": "process_terminate", "edr": "CrowdStrike/SentinelOne"}

    def _handle_suspend_suspicious_process(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR API: suspend (SIGSTOP) suspicious process "
            f"on {host} pending analysis"
        )
        return {"host": host, "action": "process_suspend", "edr": "CrowdStrike"}

    def _handle_quarantine_malicious_file(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR / AV API (CrowdStrike/Defender): "
            f"move malicious file to quarantine vault on {host}"
        )
        return {"host": host, "action": "file_quarantine", "vault": "EDR_quarantine"}

    def _handle_rollback_filesystem_changes(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call backup API (Veeam/Commvault): restore filesystem "
            f"on {host} to last known-good snapshot"
        )
        return {"host": host, "action": "fs_rollback", "backup_system": "Veeam/Commvault"}

    def _handle_restore_from_clean_snapshot(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call hypervisor API (VMware vCenter/Hyper-V): "
            f"revert VM {host} to last clean snapshot"
        )
        return {"host": host, "action": "vm_snapshot_restore", "hypervisor": "vCenter/HyperV"}

    def _handle_run_edr_deep_scan(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call EDR API (CrowdStrike Falcon): initiate on-demand "
            f"deep scan of host {host}"
        )
        return {"host": host, "scan_type": "full_deep", "edr": "CrowdStrike_Falcon"}

    def _handle_patch_vulnerable_service(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call patch management API (WSUS/Ansible): "
            f"push critical patch to service on {host}"
        )
        return {"host": host, "action": "patch_apply", "system": "WSUS/Ansible"}

    def _handle_disable_autorun_mechanisms(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call GPO/MDM API: push policy to disable autorun/autoplay "
            f"registry keys on {host}"
        )
        return {"host": host, "action": "autorun_disable", "policy_engine": "GPO/MDM"}

    def _handle_terminate_remote_desktop_session(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call WinRM / RDP session API: logoff all RDP sessions "
            f"on host {host}"
        )
        return {"host": host, "action": "rdp_logoff", "protocol": "WinRM"}

    def _handle_sandbox_suspicious_binary(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call sandboxing API (Cuckoo/Joe Sandbox): "
            f"detonate suspicious binary from {host} in isolated environment"
        )
        return {"host": host, "action": "sandbox_detonate", "sandbox": "Cuckoo/JoeSandbox"}

    # ══════════════════════════════════════════════════════════════════════
    # DATA PROTECTION HANDLERS  (actions 40–49)
    # ══════════════════════════════════════════════════════════════════════

    def _handle_encrypt_sensitive_data_at_rest(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call storage/KMS API (AWS KMS / HashiCorp Vault): "
            "enforce encryption-at-rest policy on flagged data stores"
        )
        return {"action": "encrypt_at_rest", "kms": "Vault/AWSKMS"}

    def _handle_revoke_data_export_permissions(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call DLP / IAM API: remove data-export ACL "
            f"for user/host {src}"
        )
        return {"target": src, "action": "export_permission_revoke", "system": "DLP/IAM"}

    def _handle_block_usb_exfiltration(self, alert: dict) -> dict:
        host = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call MDM / EDR API: disable USB storage devices on {host} "
            "via GPO / CrowdStrike device control"
        )
        return {"host": host, "action": "usb_block", "system": "GPO/CrowdStrike_DeviceControl"}

    def _handle_disable_email_forwarding_rules(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call Exchange/M365 API (Graph API): "
            "remove all inbox forwarding rules for affected mailbox"
        )
        return {"action": "email_forwarding_disable", "system": "Exchange/M365_GraphAPI"}

    def _handle_watermark_sensitive_documents(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call DLP / document management API (Microsoft Purview): "
            "apply invisible digital watermark to sensitive docs"
        )
        return {"action": "watermark_apply", "system": "MicrosoftPurview/DLP"}

    def _handle_enable_dlp_policy(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call DLP solution API (Symantec DLP / Purview): "
            "activate strict data-loss-prevention policy for affected segment"
        )
        return {"action": "dlp_policy_enable", "system": "SymantecDLP/Purview"}

    def _handle_freeze_database_writes(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call DB admin API (Oracle/MSSQL): "
            "set database to READ-ONLY mode — blocks all write transactions"
        )
        return {"action": "db_freeze", "mode": "READ_ONLY", "system": "Oracle/MSSQL"}

    def _handle_backup_critical_data(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call backup API (Veeam/Commvault): "
            "trigger emergency backup job for critical data stores"
        )
        return {"action": "emergency_backup", "system": "Veeam/Commvault", "priority": "immediate"}

    def _handle_rotate_encryption_keys(self, alert: dict) -> dict:
        self.logger.info(
            "[STUB] Would call KMS API (HashiCorp Vault / AWS KMS): "
            "rotate all active encryption keys and re-encrypt data"
        )
        return {"action": "key_rotate", "kms": "Vault/AWSKMS", "scope": "all_active_keys"}

    def _handle_activate_honeypot(self, alert: dict) -> dict:
        src = alert.get("source_ip", "unknown")
        self.logger.info(
            f"[STUB] Would call honeypot orchestrator (Thinkst Canary / Canarytokens): "
            f"spin up decoy service matching traffic profile of {src}"
        )
        return {
            "target_attacker": src,
            "action":          "honeypot_activate",
            "system":          "ThinkstCanary/Canarytokens",
        }


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from dataclasses import asdict
    from action_registry import ACTION_NAMES, get_action_category

    _PASS = "\033[92m[PASS]\033[0m"
    _FAIL = "\033[91m[FAIL]\033[0m"

    executor = ActionExecutor(dry_run=True)

    # Synthetic alert covering all relevant fields used by handlers
    _alert = {
        "alert_id"        : str(uuid.uuid4()),
        "timestamp"       : datetime.utcnow().isoformat() + "Z",
        "source_ip"       : "10.0.0.99",
        "destination_ip"  : "10.0.0.1",
        "source_port"     : 4444,
        "destination_port": 443,
        "protocol"        : "TCP",
        "severity"        : "high",
        "attack_type"     : "C2_Beacon",
        "layer2_confidence": 0.93,
    }

    # High-impact action indices (need approval_granted=True to run)
    _HIGH_IMPACT = frozenset({12, 13, 14, 18, 22, 28, 30, 32, 33, 34, 46})

    print("\n" + "=" * 72)
    print("  IMMUNEX Layer 3 — ActionExecutor Smoke Test (all 50 actions)")
    print("=" * 72)

    failures: list[str] = []

    for idx in range(50):
        decision = ActionDecision(
            alert_id          = _alert["alert_id"],
            action_index      = idx,
            action_name       = ACTION_NAMES[idx],
            actions           = [idx],
            action_names      = [ACTION_NAMES[idx]],
            action_categories = [get_action_category(idx)],
            requires_approval = True,
            confidence        = 0.9,
            uncertain         = False,
            impact            = "high",
            severity          = "high",
            timestamp         = _alert["timestamp"],
            raw_q_values      = None,
        )

        # Pass approval so we exercise the handler, not the gate
        result = executor.execute(decision, _alert, approval_granted=True)

        is_exec_result = isinstance(result, ExecutionResult)
        has_output     = isinstance(result.output, dict)
        correct_status = result.status in {"simulated", "executed", "failed", "pending_approval", "partial_success"}

        ok = is_exec_result and has_output and correct_status and result.error is None

        tag = _PASS if ok else _FAIL
        print(f"{tag}  [{idx:02d}] {ACTION_NAMES[idx]:<45} "
              f"status={result.status}  t={result.execution_time_ms:.3f}ms  "
              f"pre_check={result.validation_status.get('pre_check')}")

        if not ok:
            failures.append(f"[{idx}] {ACTION_NAMES[idx]} — "
                            f"result={is_exec_result} output={has_output} "
                            f"status={correct_status} error={result.error}")

    # Also test the approval gate with a high-impact action
    gate_decision = ActionDecision(
        alert_id          = _alert["alert_id"],
        action_index      = 30,
        action_name       = ACTION_NAMES[30],
        actions           = [30],
        action_names      = [ACTION_NAMES[30]],
        action_categories = [get_action_category(30)],
        requires_approval = True,
        confidence        = 0.9,
        uncertain         = False,
        impact            = "high",
        severity          = "critical",
        timestamp         = _alert["timestamp"],
        raw_q_values      = None,
    )
    gate_result = executor.execute(gate_decision, _alert, approval_granted=False)
    gate_ok = gate_result.status == "pending_approval"
    tag = _PASS if gate_ok else _FAIL
    print(f"{tag}  [GATE] kill_malicious_process without approval → "
          f"status={gate_result.status}")
    if not gate_ok:
        failures.append("Approval gate test failed")

    print("\n" + "=" * 72)
    if not failures:
        print("  \033[92mAll 51 checks passed (50 handlers + 1 approval gate).\033[0m")
    else:
        print(f"  \033[91m{len(failures)} check(s) failed:\033[0m")
        for f in failures:
            print(f"    • {f}")
        sys.exit(1)
    print("=" * 72 + "\n")