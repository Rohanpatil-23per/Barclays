"""
Action Registry for Layer 3 (Immune Response Engine).
Contains mappings for the 50 discrete actions.
"""

ACTION_NAMES = {
    0: "do_nothing",
    1: "increase_log_verbosity",
    2: "trigger_soc_alert",
    3: "snapshot_memory",
    4: "capture_network_traffic",
    5: "enable_deep_packet_inspection",
    6: "flag_for_human_review",
    7: "escalate_to_tier2",
    8: "notify_incident_manager",
    9: "generate_soc_report",
    10: "block_source_ip",
    11: "block_destination_ip",
    12: "isolate_endpoint",
    13: "quarantine_subnet",
    14: "disable_external_routing",
    15: "block_suspicious_port",
    16: "rate_limit_source",
    17: "drop_malicious_packets",
    18: "null_route_attacker",
    19: "block_c2_domain",
    20: "revoke_user_session",
    21: "force_mfa_reauthentication",
    22: "disable_compromised_account",
    23: "reset_service_account_password",
    24: "revoke_api_key",
    25: "restrict_admin_privileges",
    26: "enforce_least_privilege",
    27: "disable_lateral_movement_path",
    28: "lock_privileged_account",
    29: "audit_active_sessions",
    30: "kill_malicious_process",
    31: "suspend_suspicious_process",
    32: "quarantine_malicious_file",
    33: "rollback_filesystem_changes",
    34: "restore_from_clean_snapshot",
    35: "run_edr_deep_scan",
    36: "patch_vulnerable_service",
    37: "disable_autorun_mechanisms",
    38: "terminate_remote_desktop_session",
    39: "sandbox_suspicious_binary",
    40: "encrypt_sensitive_data_at_rest",
    41: "revoke_data_export_permissions",
    42: "block_usb_exfiltration",
    43: "disable_email_forwarding_rules",
    44: "watermark_sensitive_documents",
    45: "enable_dlp_policy",
    46: "freeze_database_writes",
    47: "backup_critical_data",
    48: "rotate_encryption_keys",
    49: "activate_honeypot"
}

def get_action_category(action_index: int) -> str:
    """Returns the category for a given action index."""
    if 0 <= action_index <= 9:
        return "monitoring"
    elif 10 <= action_index <= 19:
        return "network"
    elif 20 <= action_index <= 29:
        return "credential"
    elif 30 <= action_index <= 39:
        return "process"
    elif 40 <= action_index <= 49:
        return "data_protection"
    return "unknown"
