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
sleep 5
if docker exec immunex_kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "  ✅ Kafka ready"
else
    echo "  ⚠️  Kafka still starting, waiting 10s..."
    sleep 10
fi
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
echo "[5/5] Starting Orchestrator (port 8000)..."
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
echo ""
echo "Demo inject  : curl -X POST http://localhost:8000/demo/inject"
echo "To stop      : ./stop_immunex.sh"
