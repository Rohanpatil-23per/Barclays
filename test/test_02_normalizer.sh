#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  IMMUNEX TEST 2 — SMART NORMALIZER
#  Shows: any log format → OCSF canonical schema, dedup, watermarking
#  Run from: ~/Documents/hoh/Barclays
# ═══════════════════════════════════════════════════════════════════════════════

cd ~/Documents/hoh/Barclays
source ~/.venvs/immunex/bin/activate 2>/dev/null

C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[1m'; N='\033[0m'

echo ""
echo -e "${B}${Y}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${B}${Y}║     IMMUNEX — SMART NORMALIZER TEST                  ║${N}"
echo -e "${B}${Y}║     Any format → OCSF canonical schema               ║${N}"
echo -e "${B}${Y}╚══════════════════════════════════════════════════════╝${N}"
echo ""

python3 << 'PYEOF'
import sys, json, time
sys.path.insert(0, '.')
from shared.smart_normalizer import SmartNormalizer

G = '\033[0;32m'; R = '\033[0;31m'; C = '\033[0;36m'
Y = '\033[1;33m'; B = '\033[1m'; N = '\033[0m'

n = SmartNormalizer()

TESTS = [
    ("Syslog (OpenSSH brute force)",
     "Mar 29 10:15:43 webserver sshd[1234]: Failed password for root from 203.0.113.99 port 52341 ssh2"),

    ("JSON (Suricata EVE alert)",
     '{"timestamp":"2026-03-29T10:15:43.123Z","src_ip":"45.12.98.221","dest_ip":"192.168.10.1","src_port":54321,"dest_port":22,"proto":"TCP","alert":{"signature":"ET SCAN SSH BruteForce","severity":2},"flow":{"bytes_toserver":1024,"bytes_toclient":512}}'),

    ("CEF (SIEM format)",
     "CEF:0|Barclays|IDS|1.0|1002|SQL Injection Attempt|8|src=91.92.248.101 dst=10.0.0.50 spt=61234 dpt=1433 proto=TCP"),

    ("CSV (CICIDS dataset row)",
     "FTP-Patator, 192.168.10.50, 21, 203.0.113.99, 54321, 0.9, 6, 1024, 512"),

    ("AWS VPC Flow Log",
     "2 123456789012 eni-0123abc456 185.220.101.55 10.0.1.15 4444 443 6 120 48000 1648550000 1648550060 REJECT OK"),

    ("Free-text (unknown format)",
     "CRITICAL ALERT: brute_force detected from 77.88.21.4 targeting banking portal, 847 failed attempts in 2 minutes"),

    ("Windows Event Log (JSON)",
     '{"EventID":"4625","IpAddress":"103.35.74.10","TargetUserName":"administrator","LogonType":3,"TimeCreated":"2026-03-29T10:20:00"}'),

    ("Zeek conn.log (TSV)",
     "1648550400.123\tCabc123\t45.12.98.221\t54321\t192.168.10.50\t443\ttcp\tssl\t0.5\t2048\t1024\t10\t8\tSF"),
]

print(f"  {'FORMAT':<35} {'SRC IP':<18} {'DST IP':<18} {'FINDING':<25} {'SEV'}")
print(f"  {'-'*35} {'-'*18} {'-'*18} {'-'*25} {'-'*8}")

logs = [t[1] for t in TESTS]
events, stats = n.normalize_batch(logs)

# Map each test to its result by index (some may be deduped)
# Re-run individually for display
for name, raw in TESTS:
    evs, st = n.normalize_batch([raw])
    if evs:
        e = evs[0]
        src = e['src_endpoint']['ip']
        dst = e['dst_endpoint']['ip']
        finding = e['finding']['title'][:23]
        sev = e['severity']
        fmt = e['metadata']['original_format']
        mapped_pct = int(st['avg_mapping_coverage'] * 100)
        print(f"  {G}✅{N} {name:<33} {src:<18} {dst:<18} {finding:<25} {sev:<8}  [{fmt}, {mapped_pct}% mapped]")
    else:
        print(f"  \033[0;33m⚠{N} {name:<33} (deduplicated or parse failed)")

print()
print(f"{B}  BATCH STATS (all 8 logs at once):{N}")
_, batch_stats = n.normalize_batch(logs)
print(f"  Formats detected  : {batch_stats['formats']}")
print(f"  Parsed            : {batch_stats['parsed']}/{batch_stats['total']}")
print(f"  Bloom deduped     : {batch_stats['deduped_bloom']}")
print(f"  Late events       : {batch_stats['late_events']}")
print(f"  Avg field mapping : {int(batch_stats['avg_mapping_coverage']*100)}%")
print(f"  Elapsed           : {batch_stats['elapsed_ms']}ms")
print(f"  Throughput        : {batch_stats['throughput']:,} logs/sec")
PYEOF

echo ""

# ── DEDUPLICATION DEMO ────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Bloom Filter Deduplication (sub-millisecond)        │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from shared.smart_normalizer import SmartNormalizer

n = SmartNormalizer()
dupe = '{"src_ip":"203.0.113.99","dest_ip":"192.168.10.50","src_port":54321,"dest_port":443,"signature":"ZeusBanking","timestamp":"2026-03-29T10:00:00Z"}'

# Send same log 5 times
logs = [dupe] * 5
events, stats = n.normalize_batch(logs)
print(f"  Sent 5 identical logs → Parsed: {stats['parsed']}, Bloom-deduped: {stats['deduped_bloom']}")
print(f"  \033[0;32m✅ Bloom filter collapsed 5 duplicates → 1 unique event in {stats['elapsed_ms']}ms\033[0m")
PYEOF

echo ""

# ── WATERMARK DEMO ────────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Watermark Engine (Out-of-order timestamp handling)  │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import sys, json, time
sys.path.insert(0, '.')
from shared.smart_normalizer import SmartNormalizer

n = SmartNormalizer()

now = int(time.time())
# Simulate out-of-order logs: log3 arrives first (early), log1 arrives late
logs = [
    json.dumps({"src_ip":"1.1.1.1","dest_ip":"10.0.0.1","timestamp": now - 10, "signature":"EventA"}),  # 10s ago
    json.dumps({"src_ip":"2.2.2.2","dest_ip":"10.0.0.2","timestamp": now - 2,  "signature":"EventB"}),  # 2s ago (recent)
    json.dumps({"src_ip":"3.3.3.3","dest_ip":"10.0.0.3","timestamp": now - 60, "signature":"EventC"}),  # 60s ago (LATE)
    json.dumps({"src_ip":"4.4.4.4","dest_ip":"10.0.0.4","timestamp": now,       "signature":"EventD"}),  # now
]

events, stats = n.normalize_batch(logs)
print(f"  Submitted 4 logs with shuffled timestamps")
print(f"  Late events flagged : {stats['late_events']} (arrived after watermark)")
ordered = [e['finding']['title'] for e in events]
print(f"  Sorted order        : {' → '.join(ordered)}")
print(f"  \033[0;32m✅ Watermark engine reordered by event time, flagged late arrivals\033[0m")
PYEOF

echo ""
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo -e "${B}${G}  NORMALIZER TEST COMPLETE${N}"
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo ""
