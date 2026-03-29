#!/bin/bash
echo "=== IMMUNEX Mesh Status ==="
for NAME_IP in "ROG:localhost" "Acer:10.0.0.2" "Lenovo:10.0.0.3" "Victus:10.0.0.4" "Pavilion:10.0.0.5"; do
    NAME="${NAME_IP%%:*}"
    IP="${NAME_IP##*:}"
    for PORT in 8001 8002 8003 8004 8005; do
        RESULT=$(curl -s --max-time 2 http://$IP:$PORT/health 2>/dev/null)
        if [ -n "$RESULT" ]; then
            DEVICE=$(echo $RESULT | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('device','?'))" 2>/dev/null)
            echo "✅ $NAME L${PORT: -1} — $DEVICE"
        fi
    done
done
echo ""
echo "=== Orchestrator ==="
curl -s http://localhost:8000/ingest/nodes | python3 -c "
import json,sys
d=json.load(sys.stdin)
for url,info in d['nodes'].items():
    status = '✅' if info.get('alive') else '❌'
    device = info.get('device','?')
    gpu = '🟢 GPU' if info.get('gpu_eligible') else '🔴 CPU'
    print(f\"{status} {url} — {device} {gpu}\")
print(f\"Live nodes: {d['live_count']}/{d['total_count']}\")
"
