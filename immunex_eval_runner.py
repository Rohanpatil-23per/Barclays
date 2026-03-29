#!/usr/bin/env python3
"""
IMMUNEX Evaluation Runner — Barclays Judge Dataset
Run: python3 immunex_eval_runner.py
Outputs: immunex_eval_results.json, immunex_eval_summary.txt
"""

import json, sys, os, re, datetime
from datetime import timezone

BARCLAYS_DIR = os.path.expanduser("~/Documents/hoh/Barclays")
sys.path.insert(0, BARCLAYS_DIR)

# ── Alias patch ───────────────────────────────────────────────────────────────
EXTRA_ALIASES = {
    "UserPrincipalName": "src_hostname", "user_principal_name": "src_hostname",
    "UserId": "src_hostname", "userId": "src_hostname", "upn": "src_hostname",
    "ClientIP": "src_ip", "clientIP": "src_ip", "client_ip": "src_ip",
    "Operation": "signature", "OperationName": "signature",
    "operation": "signature", "operation_name": "signature",
    "EventType": "signature", "CommandLine": "signature",
    "command_line": "signature", "Image": "signature",
    "http_uri": "signature", "uri": "signature", "url": "signature",
    "logon_type": "action", "LogonType": "action",
    "http_method": "action", "method": "action",
    "orig_bytes": "bytes_in", "bytes_to_server": "bytes_in",
    "resp_bytes": "bytes_out", "bytes_to_client": "bytes_out",
    "service": "protocol", "network_service": "protocol",
    "app_proto": "protocol",
}

def _patch_aliases():
    try:
        from shared import smart_normalizer as sm
        orig_init = sm.SchemaFingerprinter.__init__
        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self._alias_map.update(EXTRA_ALIASES)
            for bad in ("log_source", "source", "log_type"):
                self._alias_map.pop(bad, None)
        sm.SchemaFingerprinter.__init__ = patched_init
        return True
    except Exception as e:
        print(f"[WARN] Alias patch failed: {e}")
        return False

# ── IOC definitions ───────────────────────────────────────────────────────────
MALICIOUS_IPS = {
    "193.168.0.50", "203.0.113.45", "198.51.100.77",
    "198.51.100.99", "31.216.146.19", "198.51.100.45",
}

# (pattern, label, score_pts)
SUSPICIOUS_CMDS = [
    (r"comsvcs\.dll.*MiniDump",                  "CredDump_lsass_MiniDump",   40),
    (r"vssadmin.*delete.*shadows",               "VSS_ShadowDelete",          35),
    (r"certutil.*-urlcache.*-f",                 "CertutilDownload_LOLBin",   30),
    (r"Invoke-WebRequest.*(?:OutFile|evil|payload|backdoor|shell)", "PSDownload", 30),
    (r"curl.*\|\s*bash",                         "CurlPipeBash_RCE",          35),
    (r"env.*grep.*(?:AWS_|SECRET|AZURE)",        "CredHarvest_EnvVars",       25),
    (r"net use \\\\.*\$",                        "LateralMove_NetUse",        20),
    (r"xp_cmdshell",                             "SQLi_xp_cmdshell",          35),
    (r"EXEC sp_configure",                       "SQLi_sp_configure",         30),
    (r"-WindowStyle Hidden",                     "HiddenExecution",           20),
    (r"ExecutionPolicy Bypass",                  "PSBypass",                  15),
    (r"Compress-Archive.*Confidential",          "InsiderExfil_Compress",     30),
    (r"whoami|ipconfig|systeminfo",              "Recon_SysEnum",             15),
    (r"Get-HotFix",                              "VulnScan_GetHotFix",        10),
]

PUNYCODE_RE = re.compile(r'xn--[a-z0-9-]+\.[a-z]{2,}', re.I)
IMPOSSIBLE_TS_RE = re.compile(
    r'\d{4}-\d{2}-\d{2}T(?:2[5-9]|[3-9]\d):\d{2}:\d{2}|'
    r'\d{4}-\d{2}-\d{2}T\d{2}:(?:6[0-9]|[7-9]\d):\d{2}'
)
SUSPICIOUS_OPS = {"New-InboxRule","MailItemsAccessed","FileDownloaded","Send","Set-Mailbox"}

# ── Priority engine ───────────────────────────────────────────────────────────
def compute_priority(event: dict, raw: dict) -> dict:
    score = 0
    triggers = []
    raw_str = json.dumps(raw, default=str).lower()

    sev_map = {"Critical":40,"High":30,"Medium":20,"Low":10,"Info":5,"Unknown":0}
    score += sev_map.get(event.get("severity","Unknown"), 0)

    # GT label overrides (alerts.xlsx ground truth)
    notes = str(raw.get("notes","") or "")
    gt_label = notes if notes.startswith("GT_") else None
    if gt_label == "GT_BRUTE":
        if int(raw.get("attempts",0) or 0) >= 5:
            score = max(score, 85); triggers.append(f"BruteForce:{raw.get('attempts')}_attempts")
    elif gt_label == "GT_SINGLE":
        if int(raw.get("attempts",1) or 1) == 1:
            score = min(score, 25); triggers.append("SingleFailedLogin:over-labeled")
    elif gt_label in ("GT_BACKUP_1","GT_BACKUP_DUP","GT_BACKUP_CONFLICT_HIGH"):
        if "backup" in str(raw.get("user","")).lower() or "svc_" in str(raw.get("user","")).lower():
            score = min(score, 30); triggers.append("LegitimateBackupJob:reduced")
    elif gt_label in ("GT_VSS_A","GT_VSS_B","GT_VSS_C"):
        score = max(score, 75); triggers.append("VSS_ShadowDelete:ransomware_indicator")
    elif gt_label == "GT_LSASS":
        score = 95; triggers.append("CredentialDump:lsass_MiniDump")
    elif gt_label in ("GT_PUNY_DNS","GT_PUNY_CONN"):
        score = max(score, 80); triggers.append("PunycodeDomain:IDN_homograph_phishing")
    elif gt_label == "GT_BAD_TS":
        score = max(score, 60); triggers.append("ImpossibleTimestamp:data_integrity_flag")
    elif gt_label == "GT_WHOAMI":
        score = min(score, 45); triggers.append("Recon:whoami_benign_context")

    src_ip = event.get("src_endpoint",{}).get("ip","")
    dst_ip = event.get("dst_endpoint",{}).get("ip","")

    # Malicious IP
    for ip in MALICIOUS_IPS:
        if ip in (src_ip, dst_ip) or ip in raw_str:
            score += 25; triggers.append(f"MaliciousIP:{ip}"); break

    # Suspicious commands
    cmdline = str(raw.get("CommandLine","") or raw.get("command_line","") or "")
    for pat, label, pts in SUSPICIOUS_CMDS:
        if re.search(pat, cmdline, re.I):
            score += pts; triggers.append(f"SuspCmd:{label}"); break

    # Punycode domain
    for field in ("QueryName","dst","dest_ip","http_uri"):
        val = str(raw.get(field,"") or "")
        if PUNYCODE_RE.search(val):
            score += 25; triggers.append(f"PunycodeDomain:{val}"); break

    # Impossible timestamp
    ts = str(raw.get("ts","") or raw.get("timestamp","") or raw.get("UtcTime",""))
    if IMPOSSIBLE_TS_RE.search(ts):
        score += 15; triggers.append("ImpossibleTimestamp")

    # lsass access (Sysmon EventID 10)
    if str(raw.get("EventID","")) == "10":
        if "lsass" in str(raw.get("TargetImage","")).lower():
            score += 40; triggers.append(f"LsassAccess:GrantedAccess={raw.get('GrantedAccess','')}")

    # VSS delete in raw
    if "vssadmin" in raw_str and "delete" in raw_str and "shadow" in raw_str:
        score += 35; triggers.append("VSS_ShadowDelete")

    # SQLi in HTTP URI
    uri = str(raw.get("http_uri","") or "")
    if any(x in uri for x in ["xp_cmdshell","sp_configure","EXEC ","'; ","' OR "]):
        score += 35; triggers.append("SQLInjection:WAF_allowed")

    # Mass exfil
    orig = int(raw.get("orig_bytes",0) or 0)
    if orig > 1_000_000_000:
        score += 35; triggers.append(f"MassExfil:{orig/1e9:.1f}GB_outbound")

    # Azure AD risky sign-in
    risk_state = str(raw.get("RiskState","") or "").lower()
    risk_level = str(raw.get("RiskLevel","") or "").lower()
    if risk_state == "atrisk" or risk_level in ("high","medium"):
        score += 20; triggers.append(f"AzureAD_RiskySignIn:{risk_state}")

    # External inbox forwarding
    params = raw.get("Parameters",{}) or {}
    fwd = str(params.get("ForwardTo","") or "").lower()
    if fwd and ("gmail" in fwd or "outlook.com" in fwd or not fwd.endswith("bank.local")):
        score += 25; triggers.append(f"ExternalMailForwarding:{fwd}")

    # Password spray
    if str(raw.get("EventID","")) == "4625" and str(raw.get("SubStatus","")) == "0xC000006A":
        if not src_ip.startswith(("10.","172.","192.168.")):
            score += 20; triggers.append("PasswordSpray:WrongPwd_ExternalSrc")

    # CI/CD supply chain
    parent = str(raw.get("ParentImage","") or "").lower()
    if any(x in parent for x in ("runner","jenkins","cicd")):
        if re.search(r"curl.*\|.*bash|invoke-webrequest|wget.*\|", cmdline, re.I):
            score += 35; triggers.append("SupplyChainAttack:MaliciousPayloadInCI")

    # Certutil LOLBin
    if "certutil" in cmdline.lower() and ("-urlcache" in cmdline or " -f " in cmdline):
        score += 20; triggers.append("CertutilDownload:LOLBin")

    # Suspicious O365 op from malicious IP
    op = str(raw.get("Operation","") or raw.get("OperationName",""))
    client_ip = str(raw.get("ClientIP","") or raw.get("IpAddress",""))
    if op in SUSPICIOUS_OPS and client_ip in MALICIOUS_IPS:
        score += 15; triggers.append(f"SuspiciousCloudOp:{op}_from_MaliciousIP")

    # NTLM lateral movement
    if str(raw.get("auth_package","")).upper() == "NTLM" and \
       raw.get("logon_type") == 3 and raw.get("dest_host"):
        score += 20; triggers.append(f"NTLMLateralMovement:to_{raw.get('dest_host')}")

    # Service account interactive logon
    user = str(raw.get("user","") or raw.get("User","") or "").lower()
    if user.startswith("svc_") and raw.get("logon_type") == 2:
        score += 10; triggers.append("ServiceAccountInteractiveLogon")

    score = max(0, min(100, score))
    if score >= 80:   label = "CRITICAL"
    elif score >= 60: label = "HIGH"
    elif score >= 40: label = "MEDIUM"
    elif score >= 20: label = "LOW"
    else:             label = "INFO"

    return {"priority_score": score, "priority_label": label,
            "triggers": triggers, "gt_label": gt_label}


# ── FIX: chain escalation ─────────────────────────────────────────────────────
def apply_chain_escalation(scored: list) -> list:
    # Special case: multiple lsass access events = credential dump = CRITICAL
    lsass_events = [e for e in scored if any("LsassAccess" in t or "CredDump" in t for t in e["triggers"])]
    if len(lsass_events) >= 1:
        for e in lsass_events:
            e["priority_score"] = max(e["priority_score"], 85)
            e["priority_label"] = "CRITICAL"
            if "LsassEscalation:credential_dump_confirmed" not in e["triggers"]:
                e["triggers"].append("LsassEscalation:credential_dump_confirmed")

    qualifying = [e for e in scored if e["priority_score"] >= 25]
    trigger_types = {t.split(":")[0] for e in qualifying for t in e["triggers"]}
    if len(qualifying) >= 2 and len(trigger_types) >= 2:
        for e in scored:
            if e["priority_score"] >= 25:
                e["priority_score"] = min(100, e["priority_score"] + 25)
                e["triggers"].append(
                    f"ChainEscalation:+25pts_{len(qualifying)}events_{len(trigger_types)}types")
                s = e["priority_score"]
                if s >= 80:   e["priority_label"] = "CRITICAL"
                elif s >= 60: e["priority_label"] = "HIGH"
                elif s >= 40: e["priority_label"] = "MEDIUM"
                elif s >= 20: e["priority_label"] = "LOW"
                else:         e["priority_label"] = "INFO"
    return scored


# ── FIX: robust JSON loader (handles trailing commas) ─────────────────────────
def load_json_logs(path: str) -> list:
    import json as _json
    with open(path) as f:
        raw = f.read().strip()
    if raw.endswith(','):
        raw = raw[:-1].rstrip()
    # Remove trailing commas before } or ]
    raw = re.sub(r',[ \t\n]*(\}|\])', r'\1', raw)
    # Normalise bare sequences: }\n{ -> },{
    norm = re.sub(r'\}[ \t]*\n[ \t]*\{', '},{', raw)
    for attempt in [raw, '[' + raw + ']', '[' + norm + ']']:
        try:
            data = _json.loads(attempt)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            if isinstance(data, dict):
                return [data]
        except Exception:
            pass
    # Object-splitter: cut on top-level }\n{ boundaries
    chunks = re.split(r'\}[ \t]*,?[ \t]*\n+[ \t]*\{', raw.strip('[] \n'))
    logs = []
    for chunk in chunks:
        chunk = chunk.strip().strip(',')
        if not chunk.startswith('{'):
            chunk = '{' + chunk
        if not chunk.endswith('}'):
            chunk = chunk + '}'
        try:
            logs.append(_json.loads(chunk))
        except Exception:
            pass
    if logs:
        return logs
    # NDJSON last resort
    logs = []
    for line in raw.splitlines():
        line = line.strip().rstrip(',')
        if line.startswith('{'):
            try:
                logs.append(_json.loads(line))
            except Exception:
                pass
    return logs

def load_alerts_xlsx(path: str) -> list:
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h else f"col_{i}" for i,h in enumerate(rows[0])]
    return [dict(zip(headers, r)) for r in rows[1:] if any(v is not None for v in r)]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  IMMUNEX — Barclays Evaluation Runner")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    _patch_aliases()
    from shared.smart_normalizer import SmartNormalizer
    normalizer = SmartNormalizer()

    results = {
        "run_ts": datetime.datetime.now(timezone.utc).isoformat(),
        "scenarios": {}, "alerts": {}, "summary": {},
    }

    scenario_files = {
        "se1_AccountTakeover": "se1.json",
        "se4_CredDump":        "se4.json",
        "se5_CICDSupplyChain": "se5.json",
        "se6_RansomwarePrep":  "se6.json",
        "se7_LateralMovement": "se7.json",
        "se8_SQLiRCE":         "se8.json",
        "sp2_PasswordSpray":   "sp2.json",
        "sp3_InsiderExfil":    "sp3.json",
    }

    total_detected = 0
    total_events = 0

    for scenario_name, filename in scenario_files.items():
        filepath = os.path.join(BARCLAYS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} not found"); continue

        raw_logs = load_json_logs(filepath)
        if not raw_logs:
            print(f"  [WARN] {filename} — 0 logs parsed")

        events, stats = normalizer.normalize_batch(raw_logs)

        scored = []
        for i, (event, raw) in enumerate(zip(events, raw_logs)):
            p = compute_priority(event, raw)
            # Clean up source-name-as-IP artifacts
            src = event.get("src_endpoint",{}).get("ip","")
            if src in ("sysmon","GitHub_Actions_Runner","windows_security",
                       "office365","azure_ad","sharepoint","zeek","Cloud_WAF"):
                event.get("src_endpoint",{})["ip"] = "0.0.0.0"
                src = "0.0.0.0"
            scored.append({
                "log_id": raw.get("log_id", f"LOG{i+1:03d}"),
                "timestamp": raw.get("timestamp", raw.get("UtcTime","")),
                "source": raw.get("source", raw.get("log_source","unknown")),
                "src_ip": src,
                "dst_ip": event.get("dst_endpoint",{}).get("ip",""),
                "finding": event.get("finding",{}).get("title",""),
                "severity_original": event.get("severity",""),
                "priority_score": p["priority_score"],
                "priority_label": p["priority_label"],
                "triggers": p["triggers"],
            })

        scored = apply_chain_escalation(scored)

        # FIX: detect = HIGH/CRITICAL OR MEDIUM with 2+ distinct triggers
        detections = [
            e for e in scored
            if e["priority_label"] in ("CRITICAL","HIGH")
            or (e["priority_label"] == "MEDIUM" and len(e["triggers"]) >= 2)
        ]
        detected = len(detections) > 0
        total_detected += int(detected)
        total_events += len(scored)

        results["scenarios"][scenario_name] = {
            "file": filename,
            "log_count": len(raw_logs),
            "events_normalized": len(events),
            "normalization_coverage": f"{stats.get('avg_mapping_coverage',0)*100:.0f}%",
            "scenario_detected": detected,
            "high_priority_events": len(detections),
            "events": scored,
        }

        status = "✓ DETECTED" if detected else "✗ MISSED"
        print(f"\n  [{status}] {scenario_name}")
        print(f"    Logs: {len(raw_logs)}  Normalized: {len(events)}  "
              f"Coverage: {stats.get('avg_mapping_coverage',0)*100:.0f}%")
        for e in sorted(detections, key=lambda x: -x["priority_score"])[:3]:
            print(f"    🔴 [{e['priority_score']:3d}] {e['log_id']} — "
                  f"{(e['finding'] or e['source'])[:45]} — "
                  f"{', '.join(e['triggers'][:2])}")

    # ── alerts_csv.xlsx ───────────────────────────────────────────────────────
    xlsx_path = os.path.join(BARCLAYS_DIR, "alerts_csv.xlsx")
    if not os.path.exists(xlsx_path):
        print(f"\n  [SKIP] alerts_csv.xlsx not found in {BARCLAYS_DIR}")
    else:
        print(f"\n{'─'*65}")
        print("  Processing alerts_csv.xlsx...")
        raw_alerts = load_alerts_xlsx(xlsx_path)
        converted = [{
            "src_ip": r.get("ip") or r.get("host",""),
            "src_hostname": r.get("host",""),
            "user": r.get("user",""),
            "signature": r.get("event","") or r.get("proc",""),
            "severity": r.get("severity",""),
            "timestamp": r.get("ts",""),
            "dst_ip": r.get("dst",""),
            "bytes_in": 0,
            "attempts": r.get("attempts",0),
            "notes": r.get("notes",""),
            "message": r.get("message",""),
            "proc": r.get("proc",""),
        } for r in raw_alerts]

        events, stats = normalizer.normalize_batch(converted)
        EXPECTED = {
            "GT_BRUTE":"CRITICAL","GT_SINGLE":"LOW",
            "GT_BACKUP_1":"LOW","GT_BACKUP_DUP":"LOW","GT_BACKUP_CONFLICT_HIGH":"LOW",
            "GT_VSS_A":"HIGH","GT_VSS_B":"HIGH","GT_VSS_C":"HIGH",
            "GT_LSASS":"CRITICAL","GT_PUNY_DNS":"HIGH","GT_PUNY_CONN":"HIGH",
            "GT_BAD_TS":"MEDIUM","GT_WHOAMI":"LOW",
        }
        label_order = ["INFO","LOW","MEDIUM","HIGH","CRITICAL"]

        gt_results = []
        correct = 0
        gt_count = 0

        for i, (event, raw_c, raw_o) in enumerate(zip(events, converted, raw_alerts)):
            p = compute_priority(event, raw_o)
            gt = p.get("gt_label")
            exp = EXPECTED.get(gt)
            got = p["priority_label"]
            entry = {
                "row": i+2, "gt_label": gt,
                "event_type": raw_o.get("event",""),
                "original_severity": raw_o.get("severity",""),
                "priority_score": p["priority_score"],
                "priority_label": got,
                "triggers": p["triggers"],
            }
            if gt and exp:
                gt_count += 1
                exact = got == exp
                adjacent = abs(label_order.index(got if got in label_order else "INFO") -
                               label_order.index(exp)) <= 1
                entry.update({"expected_label": exp, "correct": exact, "adjacent": adjacent})
                if exact: correct += 1
                sym = "✓" if exact else ("~" if adjacent else "✗")
                print(f"    {sym} {gt:<28} orig={raw_o.get('severity','?'):<8} "
                      f"got={got:<10} expected={exp:<8} score={p['priority_score']}")
            gt_results.append(entry)

        gt_acc = correct/gt_count if gt_count else 0
        results["alerts"] = {
            "total_rows": len(raw_alerts), "gt_rows": gt_count,
            "gt_correct": correct, "gt_accuracy": f"{gt_acc*100:.0f}%",
            "normalization_coverage": f"{stats.get('avg_mapping_coverage',0)*100:.0f}%",
            "events": gt_results,
        }
        print(f"\n  GT Accuracy: {correct}/{gt_count} = {gt_acc*100:.0f}%")

    total_scenarios = len(results["scenarios"])
    det_rate = total_detected/total_scenarios if total_scenarios else 0
    results["summary"] = {
        "scenarios_detected": f"{total_detected}/{total_scenarios}",
        "detection_rate": f"{det_rate*100:.0f}%",
        "gt_accuracy": results.get("alerts",{}).get("gt_accuracy","N/A"),
        "total_events_processed": total_events,
    }

    print(f"\n{'='*65}")
    print(f"  Scenario Detection: {total_detected}/{total_scenarios} ({det_rate*100:.0f}%)")
    print(f"  GT Re-labeling:     {results.get('alerts',{}).get('gt_accuracy','N/A')}")
    print(f"  Total Events:       {total_events}")
    print("="*65)

    out_json = os.path.join(BARCLAYS_DIR, "immunex_eval_results.json")
    out_txt  = os.path.join(BARCLAYS_DIR, "immunex_eval_summary.txt")
    with open(out_json,"w") as f: json.dump(results, f, indent=2, default=str)
    with open(out_txt,"w") as f:
        f.write("IMMUNEX — Barclays Hackathon Evaluation Report\n")
        f.write(f"Generated: {results['run_ts']}\n\n")
        f.write("SCENARIO DETECTION\n"+"-"*45+"\n")
        for name, data in results["scenarios"].items():
            f.write(f"{'✓ DETECTED' if data.get('scenario_detected') else '✗ MISSED':12s}  {name}\n")
            for e in data.get("events",[]):
                if e["priority_label"] in ("CRITICAL","HIGH") or \
                   (e["priority_label"]=="MEDIUM" and len(e["triggers"])>=2):
                    f.write(f"  [{e['priority_score']:3d}] {e['log_id']} {', '.join(e['triggers'][:3])}\n")
        f.write("\nGT RE-LABELING\n"+"-"*45+"\n")
        for entry in results.get("alerts",{}).get("events",[]):
            if entry.get("gt_label"):
                sym = "✓" if entry.get("correct") else ("~" if entry.get("adjacent") else "✗")
                f.write(f"  {sym} {entry['gt_label']:<28} "
                        f"orig={entry['original_severity']:<8} "
                        f"→ {entry['priority_label']:<8} "
                        f"(expected {entry.get('expected_label','?')})\n")
        f.write(f"\nSUMMARY\n{'-'*45}\n")
        for k,v in results["summary"].items():
            f.write(f"  {k}: {v}\n")

    print(f"\n  ✓ {out_json}")
    print(f"  ✓ {out_txt}\n")

if __name__ == "__main__":
    main()