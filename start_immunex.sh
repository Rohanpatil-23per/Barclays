#!/bin/bash
cd "$(dirname "$0")"

VENV="$HOME/.venvs/immunex/bin"
LOG="logs"
mkdir -p "$LOG"

# Layer URLs — override with env vars when distributed
L2_URL="${LAYER2_URL:-http://localhost:8002}"
L3_URL="${LAYER3_URL:-http://localhost:8003}"
L4_URL="${LAYER4_URL:-http://localhost:8004}"


echo "╔══════════════════════════════════════╗"
echo "║     IMMUNEX STARTUP SEQUENCE         ║"
echo "╚══════════════════════════════════════╝"

# ── Step 1: Kill all previous layer processes ────────────────────────────────
echo ""
echo "[1/6] Killing previous processes..."
fuser -k 8000/tcp 8001/tcp 8002/tcp 8003/tcp 8004/tcp 2>/dev/null
pkill -f "uvicorn.*immunex" 2>/dev/null
pkill -f "uvicorn.*layer" 2>/dev/null
pkill -f "uvicorn.*orchestrator" 2>/dev/null
pkill -f "uvicorn.*main:app" 2>/dev/null
sleep 2
echo "  ✅ Previous processes cleared"

# ── Step 2: Fresh Docker restart ─────────────────────────────────────────────
echo ""
echo "[2/6] Fresh Docker restart (wiping stale state)..."
docker compose down -v --remove-orphans 2>/dev/null
sleep 3
docker compose up -d
echo "  ⏳ Waiting for Kafka (up to 60s)..."
for i in $(seq 1 20); do
    if nc -z localhost 9092 2>/dev/null; then
        echo "  ✅ Kafka ready (${i}x3s)"
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "  ⚠️  Kafka slow — continuing anyway (non-fatal)"
        break
    fi
    sleep 3
done

# Wait for ES and Redis
for i in $(seq 1 10); do
    nc -z localhost 9200 2>/dev/null && break
    sleep 2
done
echo "  ✅ Elasticsearch ready"

for i in $(seq 1 10); do
    nc -z localhost 6379 2>/dev/null && break
    sleep 1
done
echo "  ✅ Redis ready"

# ── Step 3: Ollama ────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Starting Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Ollama already running"
else
    ollama serve > "$LOG/ollama.log" 2>&1 &
    sleep 5
    echo "  ✅ Ollama started"
fi

# ── Step 4: CUDA ──────────────────────────────────────────────────────────────
echo ""
echo "[4/6] CUDA environment..."
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
source "$HOME/.venvs/immunex/bin/activate"
CUDA=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "  ✅ Device: $CUDA"

# ── Step 5: Start local layers ────────────────────────────────────────────────
echo ""
echo "[5/6] Starting local layers..."

# Helper: start a layer and wait for health
start_layer() {
    local name="$1"
    local cmd="$2"
    local port="$3"
    local logfile="$4"
    local wait="${5:-15}"

    eval "$cmd > $logfile 2>&1 &"
    local pid=$!
    echo "  ⏳ $name starting (PID $pid)..."
    for i in $(seq 1 $wait); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✅ $name running on port $port"
            return 0
        fi
        sleep 1
    done
    echo "  ❌ $name failed — check $logfile"
    return 1
}

# Layer 1 — always local
start_layer "Layer 1" \
    "$VENV/uvicorn layer1_detection.server:app --host 0.0.0.0 --port 8001" \
    8001 "$LOG/layer1.log" 30

# Layers 2-5: only start locally if URL is localhost
if [[ "$L2_URL" == *"localhost"* ]]; then
    start_layer "Layer 2" \
        "$VENV/uvicorn layer2_correlation.server:app --host 0.0.0.0 --port 8002" \
        8002 "$LOG/layer2.log" 20
else
    echo "  ↗  Layer 2 remote: $L2_URL"
fi

if [[ "$L3_URL" == *"localhost"* ]]; then
    (cd "Layer3 Response Engine/Response_engine" && \
     $VENV/uvicorn main:app --host 0.0.0.0 --port 8003 > "../../$LOG/layer3.log" 2>&1 &)
    sleep 12
    curl -s http://localhost:8003/health > /dev/null 2>&1 \
        && echo "  ✅ Layer 3 running on port 8003" \
        || echo "  ❌ Layer 3 failed — check $LOG/layer3.log"
else
    echo "  ↗  Layer 3 remote: $L3_URL"
fi

if [[ "$L4_URL" == *"localhost"* ]]; then
    start_layer "Layer 4" \
        "$VENV/uvicorn layer4_immunity.server:app --host 0.0.0.0 --port 8004" \
        8004 "$LOG/layer4.log" 15
else
    echo "  ↗  Layer 4 remote: $L4_URL"
fi



# ── Step 6: Orchestrator ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting Orchestrator..."
LAYER2_URL="$L2_URL" \
LAYER3_URL="$L3_URL" \
LAYER4_URL="$L4_URL" \

$VENV/uvicorn orchestrator.server:app --host 0.0.0.0 --port 8000 > "$LOG/orchestrator.log" 2>&1 &
sleep 8
curl -s http://localhost:8000/health > /dev/null 2>&1 \
    && echo "  ✅ Orchestrator running on port 8000" \
    || echo "  ❌ Orchestrator failed — check $LOG/orchestrator.log"

# ── Final status ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║         IMMUNEX STATUS               ║"
echo "╚══════════════════════════════════════╝"
curl -s http://localhost:8000/health 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  Orchestrator : {d[\"status\"]}')
    for layer, status in d['layers'].items():
        icon = '✅' if status else '⭕'
        label = 'online' if status else 'offline'
        print(f'  {icon} {layer}: {label}')
except:
    print('  ⚠️  Could not reach orchestrator')
" 2>/dev/null

echo ""
echo "Endpoints:"
echo "  Orchestrator : http://localhost:8000"
echo "  Layer 1      : http://localhost:8001"
echo "  Layer 2      : $L2_URL"
echo "  Layer 3      : $L3_URL"
echo "  Layer 4      : $L4_URL"
echo ""
echo "Distributed run:"
echo "  LAYER2_URL=http://10.0.0.2:8002 LAYER3_URL=http://10.0.0.3:8003 \\"
echo "  LAYER4_URL=http://10.0.0.4:8004 \\"
echo "  ./start_immunex.sh"
echo ""
echo "Demo inject  : curl -X POST http://localhost:8000/demo/inject"
echo "To stop      : ./stop_immunex.sh"
