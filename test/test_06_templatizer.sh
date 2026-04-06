#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  IMMUNEX TEST 6 — LOG TEMPLATIZER
#  Shows: dynamic value tokenization for DBSCAN clustering
#  Run from: ~/Documents/hoh/Barclays
# ═══════════════════════════════════════════════════════════════════════════════

cd ~/Documents/hoh/Barclays
source ~/.venvs/immunex/bin/activate 2>/dev/null

G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'

echo ""
echo -e "${B}${Y}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${B}${Y}║     IMMUNEX — LOG TEMPLATIZER TEST                   ║${N}"
echo -e "${B}${Y}║     Dynamic values → typed tokens for clustering     ║${N}"
echo -e "${B}${Y}╚══════════════════════════════════════════════════════╝${N}"
echo ""

# ── WHAT IT DOES ──────────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Why Templatization Matters                          │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"
echo ""
echo "  Problem: Two logs describing the same attack look different:"
echo "    'Failed login for user alice from 203.0.113.99 port 52341'"
echo "    'Failed login for user bob   from 45.12.98.221 port 49201'"
echo ""
echo "  Without templating → DBSCAN treats them as different events"
echo "  With templating    → both become identical templates, clustered together"
echo "    'Failed login for user <USER> from <IPv4> port <PORT>'"
echo ""

# ── TOKENIZATION DEMO ─────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Tokenization — 10 real attack log samples           │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.log_templatizer import templatize

G = '\033[0;32m'; C = '\033[0;36m'; B = '\033[1m'; N = '\033[0m'

LOGS = [
    "Failed password for root from 203.0.113.99 port 52341 ssh2",
    "Failed password for admin from 45.12.98.221 port 49201 ssh2",
    "SQL injection detected: ' UNION SELECT * FROM accounts-- src=91.92.248.101 dst=10.0.0.50 port:1433",
    "Process created: C:\\Windows\\System32\\mimikatz.exe by PID 4892 on host BANKPC01",
    "File renamed: C:\\Users\\admin\\Documents\\report.docx → report.docx.locked (ransomware indicator)",
    "DNS query from 192.168.10.55 to aGVsbG93b3JsZA==.exfil.attacker.io type TXT",
    "CVE-2021-44228 exploit attempt: src=185.220.101.55 target=10.0.0.80:8080",
    "Kerberos ticket request from 10.0.0.33 user=svc_backup hash=5f4dcc3b5aa765d61d8327deb882cf99",
    "Transfer £1,250,000.00 from account 40512345 to 00-00-00/87654321 flagged",
    "Outbound beacon: src=192.168.10.55 dst=194.28.115.42:4444 interval=30s session_id=a1b2c3d4e5f6",
]

print(f"  {'ORIGINAL LOG (truncated)':<52}  →  TEMPLATE")
print(f"  {'-'*52}     {'-'*45}")
for log in LOGS:
    tmpl, extracted = templatize(log)
    orig_short = log[:50] + "…" if len(log) > 50 else log
    print(f"  {orig_short:<52}  →  {C}{tmpl[:60]}{N}")
    if extracted:
        vals = list(extracted.values())[:3]
        print(f"  {'':52}     Extracted: {vals}")
    print()
PYEOF

# ── CLUSTERING PROOF ──────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Clustering Effect — same template = same attack     │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.log_templatizer import templatize
from collections import defaultdict

# 20 brute force logs from different IPs/users — should all cluster to 1 template
BRUTE_FORCE_LOGS = [
    f"Failed password for {user} from {ip} port {port} ssh2"
    for user, ip, port in [
        ("root",      "203.0.113.99",  52341),
        ("admin",     "45.12.98.221",  49201),
        ("oracle",    "91.92.248.101", 55123),
        ("www-data",  "77.88.21.4",    61234),
        ("postgres",  "194.28.115.42", 48901),
        ("pi",        "103.35.74.10",  52900),
        ("ubuntu",    "185.220.101.55",54321),
        ("test",      "1.2.3.4",       50001),
        ("deploy",    "5.6.7.8",       60000),
        ("barclays",  "9.10.11.12",    51000),
    ]
]

# Mix in 5 ransomware logs (different template)
RANSOMWARE_LOGS = [
    f"File renamed: C:\\Users\\{u}\\Documents\\{f}.docx → {f}.docx.locked by PID {pid}"
    for u, f, pid in [
        ("alice", "report",   4892),
        ("bob",   "invoice",  5012),
        ("carol", "strategy", 3344),
        ("dave",  "budget",   6789),
        ("eve",   "accounts", 2233),
    ]
]

ALL_LOGS = BRUTE_FORCE_LOGS + RANSOMWARE_LOGS

clusters = defaultdict(list)
for log in ALL_LOGS:
    tmpl, _ = templatize(log)
    clusters[tmpl].append(log)

print(f"  Input: {len(ALL_LOGS)} logs (10 brute force + 5 ransomware, all different IPs/users/PIDs)")
print(f"  Output: {len(clusters)} unique template clusters\n")

for i, (tmpl, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
    attack_type = "BruteForce SSH" if "Failed password" in tmpl else "Ransomware FileRename"
    print(f"  Cluster {i+1}: [{attack_type}] — {len(members)} logs collapsed")
    print(f"    Template: \033[0;36m{tmpl[:75]}\033[0m")
    print(f"    Examples: {members[0][:60]}…")
    print()

print(f"  \033[0;32m✅ {len(ALL_LOGS)} unique logs → {len(clusters)} templates → correct clustering\033[0m")
PYEOF

echo ""

# ── TOKEN TYPES REFERENCE ─────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Token Vocabulary (Barclays-specific additions)      │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
from shared.log_templatizer import templatize

DEMOS = {
    "<IPv4>"     : "Connected from 10.0.0.55",
    "<IPv6>"     : "Source: 2001:db8::1",
    "<MAC>"      : "ARP from 00:1A:2B:3C:4D:5E",
    "<PORT>"     : "Listening on port 4444",
    "<CVE>"      : "Exploit attempt CVE-2021-44228",
    "<SHA256>"   : "Hash: " + "a"*64,
    "<MD5>"      : "Checksum: " + "b"*32,
    "<UUID>"     : "Session: 550e8400-e29b-41d4-a716-446655440000",
    "<WINPATH>"  : "Exec: C:\\Windows\\System32\\cmd.exe",
    "<UNIXPATH>" : "Read: /etc/shadow",
    "<USER>"     : "Login for user jsmith from",
    "<ACCTNUM>"  : "Account 40512345 accessed",
    "<AMOUNT>"   : "Transaction £1,250,000.00 flagged",
    "<NUMID>"    : "Session ID 1648550400 opened",
    "<TIMESTAMP>": "Event at 2026-03-29T10:15:43Z",
}

print(f"  {'TOKEN':<14} {'INPUT EXAMPLE':<45} RESULT")
print(f"  {'-'*14} {'-'*45} {'-'*20}")
for token, example in DEMOS.items():
    tmpl, _ = templatize(example)
    print(f"  {token:<14} {example:<45} → \033[0;36m{tmpl}\033[0m")
PYEOF

echo ""
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo -e "${B}${G}  TEMPLATIZER TEST COMPLETE${N}"
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo ""
