#!/bin/bash
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════╗"
echo "║           IMMUNEX DEMO — ZEUS BANKING TROJAN     ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Start everything if not already running ──────────────────────────────────
ORCH_UP=$(curl -s http://localhost:8000/health 2>/dev/null | grep -c '"status":"ok"')
if [ "$ORCH_UP" -eq 0 ]; then
    echo "⚡ Starting IMMUNEX..."
    ./start_immunex.sh
else
    echo "✅ IMMUNEX already running"
fi

echo ""
echo "🦠 Injecting Zeus Banking Trojan alert..."
echo ""

RESULT=$(curl -s -X POST http://localhost:8000/demo/inject)

echo "$RESULT" | python3 - << 'EOF'
import sys, json

d = json.loads(sys.stdin.read())

print(f"  Alert ID      : {d.get('pipeline_id','?')}")
print(f"  Source IP     : 203.0.113.99  →  192.168.10.50")
print(f"  Attack        : Zeus Banking Trojan")
print(f"  Verdict       : {'🔴 ' if d.get('verdict')=='ANOMALOUS' else '🟢 '}{d.get('verdict','?')}")
print(f"  Anomaly Score : {d.get('anomaly_score', '?')}")
print()

layers = {
    'layer1': 'Innate Detection     (RoBERTa + IsoForest)',
    'layer2': 'Adaptive Correlation (Attack Graph)',
    'layer3': 'Response Engine      (RL Policy)',
    'layer4': 'Immunity Memory      (Prediction)',
    'layer5': 'Threat Memory        (Playbook)',
}
for key, label in layers.items():
    v = d.get(key)
    if isinstance(v, dict):
        err = v.get('error') or v.get('status') == 'failed' and v.get('error','')
        if err and err is not True:
            print(f"  ⚠️  L{key[-1]} {label}")
            print(f"      └─ {str(err)[:80]}")
        else:
            print(f"  ✅ L{key[-1]} {label}")
    else:
        print(f"  ❌ L{key[-1]} {label} — offline")

print()
fa = d.get('final_action') or {}
if fa:
    print(f"  Final Action  : {fa.get('action') or fa.get('decision','?')}")
pb = d.get('playbook') or {}
steps = pb.get('steps') or pb.get('predicted_threats') or []
if steps:
    print(f"  Playbook      : {len(steps)} step(s) generated")

err = d.get('error')
if err:
    print(f"\n  ⚠️  Pipeline error: {err}")
EOF

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Demo complete. Run again: ./demo.sh             ║"
echo "╚══════════════════════════════════════════════════╝"
