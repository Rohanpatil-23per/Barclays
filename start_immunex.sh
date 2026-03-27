#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV="$HOME/.venvs/immunex/bin/activate"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════╗"
echo "║     IMMUNEX STARTUP SEQUENCE         ║"
echo "╚══════════════════════════════════════╝"

echo ""
echo "[1/5] Starting Docker infrastructure (Kafka, ES, Redis)..."
docker compose up -d
echo "  ⏳ Waiting for Kafka..."
for i in $(seq 1 20); do
    if nc -z localhost 9092 2>/dev/null; then
        echo "  ✅ Kafka ready"
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "  ❌ Kafka failed to start — check: docker logs immunex_kafka"
        exit 1
    fi
    sleep 3
done
echo "  ✅ Elasticsearch (port 9200)"
echo "  ✅ Redis (port 6379)"

echo ""
echo "[2/5] Starting Ollama (llama3.1:8b)..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Ollama already running"
else
    ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 5
    echo "  ✅ Ollama started"
fi

echo ""
echo "[3/5] Setting up CUDA environment..."
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
source "$VENV"
CUDA=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "  ✅ Device: $CUDA"

echo ""
echo "[4/5] Starting Layer 1 (port 8001)..."
python3 layer1_detection/server.py > "$LOG_DIR/layer1.log" 2>&1 &
echo "  ⏳ Loading models... (PID $!)"
sleep 15
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "  ✅ Layer 1 running"
else
    echo "  ❌ Layer 1 failed — check logs/layer1.log"
    exit 1
fi

echo ""
echo "[5/5] Starting Layer 2 (port 8002)..."
$HOME/.venvs/immunex/bin/uvicorn layer2_correlation.server:app --host 0.0.0.0 --port 8002 > "$LOG_DIR/layer2.log" 2>&1 &
sleep 8
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "  ✅ Layer 2 running"
else
    echo "  ❌ Layer 2 failed — check logs/layer2.log"
fi

echo ""
echo "[6/5] Starting Layer 3 (port 8003)..."
cd "Layer3 Response Engine/Response_engine"
$HOME/.venvs/immunex/bin/uvicorn main:app --host 0.0.0.0 --port 8003 > "../../$LOG_DIR/layer3.log" 2>&1 &
cd ../..
sleep 10
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "  ✅ Layer 3 running"
else
    echo "  ❌ Layer 3 failed — check logs/layer3.log"
fi

echo ""
echo "[7/5] Starting Layer 4 (port 8004)..."
$HOME/.venvs/immunex/bin/uvicorn layer4_immunity.server:app --host 0.0.0.0 --port 8004 > "$LOG_DIR/layer4.log" 2>&1 &
sleep 8
if curl -s http://localhost:8004/health > /dev/null 2>&1; then
    echo "  ✅ Layer 4 running"
else
    echo "  ❌ Layer 4 failed — check logs/layer4.log"
fi

echo ""
echo "[8/5] Starting Layer 5 (port 8005)..."
$HOME/.venvs/immunex/bin/uvicorn "Layer5_Threat Memory.server":app --host 0.0.0.0 --port 8005 > "$LOG_DIR/layer5.log" 2>&1 &
sleep 6
if curl -s http://localhost:8005/health > /dev/null 2>&1; then
    echo "  ✅ Layer 5 running"
else
    echo "  ❌ Layer 5 failed — check logs/layer5.log"
fi

echo ""
echo "[9/5] Starting Orchestrator (port 8000)..."
python3 orchestrator/server.py > "$LOG_DIR/orchestrator.log" 2>&1 &
sleep 5
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ Orchestrator running"
else
    echo "  ❌ Orchestrator failed — check logs/orchestrator.log"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         IMMUNEX STATUS               ║"
echo "╚══════════════════════════════════════╝"
curl -s http://localhost:8000/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Orchestrator : {d[\"status\"]}')
for layer, status in d['layers'].items():
    icon = '✅' if status else '⭕'
    label = 'online' if status else 'offline'
    print(f'  {icon} {layer}: {label}')
"
echo ""
echo "Endpoints:"
echo "  Orchestrator : http://localhost:8000"
echo "  Layer 1      : http://localhost:8001"
echo "  Layer 2      : http://localhost:8002"
echo "  Layer 3      : http://localhost:8003"
echo "  Layer 4      : http://localhost:8004"
echo "  Layer 5      : http://localhost:8005"
echo ""
echo "Demo inject  : curl -X POST http://localhost:8000/demo/inject"
echo "To stop      : ./stop_immunex.sh"
