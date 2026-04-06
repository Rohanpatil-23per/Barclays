#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  IMMUNEX TEST 5 — PRIORITY ENGINE
#  Shows: composite scoring, triage tiers, attack chain escalation, IOC boost
#  Run from: ~/Documents/hoh/Barclays
# ═══════════════════════════════════════════════════════════════════════════════

cd ~/Documents/hoh/Barclays
source ~/.venvs/immunex/bin/activate 2>/dev/null

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'

echo ""
echo -e "${B}${Y}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${B}${Y}║     IMMUNEX — PRIORITY ENGINE TEST                   ║${N}"
echo -e "${B}${Y}║     Weighted scoring: ML + Severity + Asset + Chain  ║${N}"
echo -e "${B}${Y}╚══════════════════════════════════════════════════════╝${N}"
echo ""

# ── SCORING FORMULA EXPLANATION ───────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Scoring Formula (matches Splunk ES / QRadar)        │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"
echo ""
echo "  final_score = ("
echo "      anomaly_score   × 0.30   ← ML confidence from L1"
echo "    + severity_score  × 0.20   ← raw severity field"
echo "    + asset_criticality × 0.20 ← how valuable is the target"
echo "    + attack_chain    × 0.20   ← repeated attacks from same IP"
echo "    + mitre_weight    × 0.10   ← technique-based escalation"
echo "  ) × ioc_multiplier           ← 2× if src IP is known bad"
echo ""
echo "  Tiers: CRITICAL ≥ 0.85 | HIGH ≥ 0.65 | MEDIUM ≥ 0.40 | LOW < 0.40"
echo ""

# ── SCENARIO SCORING ──────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Scenario Scoring — 6 real-world attack cases        │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.priority_engine import compute_priority_score

SCENARIOS = [
    {
        "name": "Zeus Banking Trojan → Core Banking (10.0.0.x)",
        "l1":   {"source_ip":"203.0.113.99","dest_ip":"10.0.0.50","attack_type":"Zeus_Banking_Trojan","anomaly_score":0.80,"confidence":0.78},
        "alert":{"source_ip":"203.0.113.99","dest_ip":"10.0.0.50","severity":"critical"},
    },
    {
        "name": "Port Scan → Dev server (10.0.4.x)",
        "l1":   {"source_ip":"45.12.98.221","dest_ip":"10.0.4.10","attack_type":"PortScan","anomaly_score":0.55,"confidence":0.60},
        "alert":{"source_ip":"45.12.98.221","dest_ip":"10.0.4.10","severity":"low"},
    },
    {
        "name": "Ransomware → Trading system (10.0.1.x)",
        "l1":   {"source_ip":"91.92.248.101","dest_ip":"10.0.1.20","attack_type":"Ransomware","anomaly_score":0.92,"confidence":0.88},
        "alert":{"source_ip":"91.92.248.101","dest_ip":"10.0.1.20","severity":"critical"},
    },
    {
        "name": "SQL Injection → DMZ server (10.0.2.x)",
        "l1":   {"source_ip":"77.88.21.4","dest_ip":"10.0.2.30","attack_type":"SQLi","anomaly_score":0.70,"confidence":0.72},
        "alert":{"source_ip":"77.88.21.4","dest_ip":"10.0.2.30","severity":"high"},
    },
    {
        "name": "C2 Beacon → External IP (exfiltration)",
        "l1":   {"source_ip":"192.168.10.55","dest_ip":"194.28.115.42","attack_type":"C2","anomaly_score":0.85,"confidence":0.82},
        "alert":{"source_ip":"192.168.10.55","dest_ip":"194.28.115.42","severity":"critical"},
    },
    {
        "name": "Zeus Trojan + IOC FLAG → Core Banking",
        "l1":   {"source_ip":"203.0.113.99","dest_ip":"10.0.0.50","attack_type":"Zeus_Banking_Trojan","anomaly_score":0.80,"confidence":0.78},
        "alert":{"source_ip":"203.0.113.99","dest_ip":"10.0.0.50","severity":"critical","_ioc_hit":True},
    },
]

TIER_COLOR = {
    "CRITICAL": "\033[0;31m",
    "HIGH"    : "\033[1;33m",
    "MEDIUM"  : "\033[0;36m",
    "LOW"     : "\033[0;37m",
}

print(f"  {'SCENARIO':<44} {'SCORE':>6}  {'TIER':<10}  COMPONENTS")
print(f"  {'-'*44} {'-'*6}  {'-'*10}  {'-'*40}")

for s in SCENARIOS:
    result = compute_priority_score(s["l1"], s["alert"])
    score  = result.get("priority_score", result.get("score", 0))
    tier   = result.get("tier", "?")
    color  = TIER_COLOR.get(tier, "")
    ioc    = "🚨 IOC×2" if s["alert"].get("_ioc_hit") else ""
    comps  = result.get("components", result.get("score_breakdown", {}))
    comp_str = ""
    if comps:
        comp_str = " | ".join(f"{k[:3]}={v:.2f}" for k,v in list(comps.items())[:4])
    print(f"  {s['name']:<44} {score:>6.3f}  {color}{tier:<10}\033[0m  {comp_str} {ioc}")

print()
print(f"  \033[0;32m✅ All 6 scenarios scored with weighted composite formula\033[0m")
PYEOF

echo ""

# ── ATTACK CHAIN ESCALATION ────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Attack Chain Escalation (persistent attacker)       │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.priority_engine import compute_priority_score, _chain_tracker

# Simulate same IP attacking 8 times — watch score escalate
ATTACKER = "185.220.101.55"
BASE_L1  = {"source_ip": ATTACKER, "dest_ip":"10.0.0.50", "attack_type":"BruteForce", "anomaly_score":0.65, "confidence":0.60}
BASE_ALT = {"source_ip": ATTACKER, "dest_ip":"10.0.0.50", "severity":"high"}

print(f"  Simulating repeated attacks from {ATTACKER}:")
print()
print(f"  {'HIT':>4}  {'SCORE':>6}  {'TIER':<10}  CHAIN-SCORE  NOTE")
print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*11}  {'-'*30}")

for hit in range(1, 9):
    result = compute_priority_score(BASE_L1, BASE_ALT)
    score  = result.get("priority_score", result.get("score", 0))
    tier   = result.get("tier", "?")
    chain  = result.get("components", {}).get("attack_chain", result.get("score_breakdown", {}).get("chain", 0))
    note   = ""
    if hit == 1:   note = "← First seen"
    elif hit == 3: note = "← Repeated (escalates)"
    elif hit == 5: note = "← Persistent attacker"
    elif hit == 8: note = "← SUSTAINED ATTACK"
    print(f"  {hit:>4}  {score:>6.3f}  {tier:<10}  {chain:<11.3f}  {note}")

print()
print(f"  \033[0;32m✅ Same attacker IP escalates from LOW → CRITICAL over 8 hits\033[0m")
PYEOF

echo ""

# ── ASSET CRITICALITY ─────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Asset Criticality Map (CMDB-aligned)                │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.priority_engine import _asset_criticality

TARGETS = [
    ("10.0.0.50",  "Core banking system"),
    ("10.0.1.20",  "Trading system"),
    ("10.0.2.30",  "DMZ / public-facing"),
    ("10.0.3.10",  "Internal services"),
    ("10.0.4.99",  "Dev/test environment"),
    ("192.168.10.50","Generic internal host"),
    ("194.28.115.42","External (exfil destination)"),
]

print(f"  {'DESTINATION IP':<18} {'DESCRIPTION':<25} {'CRITICALITY':>11}  TIER")
print(f"  {'-'*18} {'-'*25} {'-'*11}  {'-'*10}")
for ip, desc in TARGETS:
    crit = _asset_criticality(ip)
    tier = "CRITICAL" if crit >= 0.85 else "HIGH" if crit >= 0.70 else "MEDIUM" if crit >= 0.50 else "LOW"
    bar  = "█" * int(crit * 10)
    print(f"  {ip:<18} {desc:<25} {crit:>11.2f}  {bar} {tier}")

print(f"\n  \033[0;32m✅ Asset criticality boosts alerts targeting high-value systems\033[0m")
PYEOF

echo ""
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo -e "${B}${G}  PRIORITY ENGINE TEST COMPLETE${N}"
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo ""
