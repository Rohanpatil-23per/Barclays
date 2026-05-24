# IMMUNEX – Multi-Layer Adaptive Threat Detection & Response System

![IMMUNEX](https://img.shields.io/badge/version-3.0.0-brightgreen) ![Python](https://img.shields.io/badge/python-3.9+-blue) ![License](https://img.shields.io/badge/license-Proprietary-red)

**IMMUNEX** is a production-grade, distributed security threat detection and automated response system designed for financial institutions. It implements a five-layer defense pipeline that detects anomalies, correlates attack chains, generates adaptive responses, maintains immunity learning, and builds long-term threat memory.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running IMMUNEX](#running-immunex)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🚀 Quick Start

### Prerequisites
- **OS**: Linux (Ubuntu 20.04+) or macOS 
- **Python**: 3.9+
- **GPU** (recommended): NVIDIA CUDA 12.6+
- **Docker**: For Kafka, Elasticsearch, Redis
- **RAM**: 16GB minimum (32GB recommended)

### Clone & Setup (5 minutes)
```bash
git clone https://github.com/yourusername/immunex.git
cd immunex
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start Full Stack
```bash
# Terminal 1: Start all services (Docker + all 5 layers)
./start_immunex.sh

# Terminal 2: Send test alert
curl -X POST http://localhost:8000/demo/inject

# Check status
curl http://localhost:8000/health
```

**Expected output**: All 5 layers online, pipeline processes alert end-to-end.

---

## 🏗️ Architecture Overview

IMMUNEX is a **five-layer distributed pipeline**. Each layer is horizontally scalable and can run on separate machines via WireGuard mesh network.

```
┌────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (8000)                        │
│  - Request routing, circuit breaking, metrics, rate limiting   │
└────────────────────────────────────────────────────────────────┘
        ↓              ↓              ↓              ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   LAYER 1    │ │   LAYER 2    │ │   LAYER 3    │ │   LAYER 4    │
│ DETECTION    │ │ CORRELATION  │ │ RESPONSE     │ │ IMMUNITY     │
│  Port 8001   │ │  Port 8002   │ │  Port 8003   │ │  Port 8004   │
│              │ │              │ │              │ │              │
│ Anomaly      │ │ Attack Chain │ │ DQN Decision │ │ LoRA+EWC     │
│ Detection    │ │ Correlation  │ │ Generation   │ │ Adaptation   │
│ GATv2 +      │ │ BiLSTM +     │ │ Z3 Safety    │ │ 77-feature   │
│ AutoEncoder  │ │ HMM History  │ │ Verification │ │ Classifier   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                                          ↓
                                ┌──────────────────┐
                                │   LAYER 5        │
                                │ THREAT MEMORY    │
                                │   Port 8005      │
                                │                  │
                                │ LSTM+HMM History │
                                │ SQLite Chains    │
                                │ Attacker Profiles│
                                └──────────────────┘
```

### Layer 1: Detection (Port 8001)
- **Purpose**: Identify anomalous logs in real-time
- **Models**: GATv2 graph neural network + AutoEncoder novelty detection
- **Input**: Raw logs (syslog, JSON, CSV, etc.)
- **Output**: Anomaly score (0-1), confidence, embedding, attack type
- **Throughput**: 10k+ logs/sec (single GPU)

### Layer 2: Correlation (Port 8002)
- **Purpose**: Link related alerts into attack chains
- **Models**: Transformer Spatial Reasoning + BiLSTM Narrative Tracker + HMM Predictor
- **Input**: L1 anomaly results + embedding
- **Output**: Attack chain ID, MITRE ATT&CK stage, confidence, predicted next stage
- **Features**: Watermarking, temporal reordering, 50-alert sequence windows

### Layer 3: Response (Port 8003)
- **Purpose**: Decide optimal remediation actions
- **Models**: Dueling DQN policy + Z3 theorem prover for safety verification
- **Input**: L2 attack graph + chain history
- **Output**: Action (do_nothing, isolate, block, monitor), confidence, rationale
- **Features**: Human-in-the-loop approval gate, compliance checks (RBI, GDPR, DORA)

### Layer 4: Immunity (Port 8004)
- **Purpose**: Learn and adapt to new threats
- **Models**: Logistic Regression with LoRA adapters + EWC (Elastic Weight Consolidation)
- **Input**: Labeled incident data (77-dim CICIDS features)
- **Output**: Threat severity classification (Benign, Low, Medium, High, Critical)
- **Features**: Continual learning without catastrophic forgetting, 95.88% accuracy on master dataset

### Layer 5: Threat Memory (Port 8005)
- **Purpose**: Build attacker profiles and predict future moves
- **Models**: LSTM + HMM state progression + SQLite persistence
- **Input**: Confirmed attack chains
- **Output**: Predicted next stage, risk escalation, attacker IP profile
- **Features**: Cross-session attack tracking, persistence across restarts

---

## 💻 System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 8+ cores | L1/L2 inference parallelizable |
| RAM | 32GB | Models: L1=2GB, L2=1.5GB, L4=1.2GB, others=0.5GB each |
| Storage | 100GB | Weights + datasets + logs |
| GPU | NVIDIA RTX 2060+ | CUDA 12.6, CuDNN 8.6+ |
| Network | 1Gbps+ | For distributed mesh nodes |
| OS | Linux/macOS | Windows requires WSL2 |

### Tested Configurations
- ✅ RTX 4070 + Ryzen 7 5700X + 64GB RAM (primary)
- ✅ RTX 4050 Laptop + 32GB RAM (Layer 2)
- ✅ RTX 3050 + 32GB RAM (Layer 3)
- ✅ RTX 2050 Laptop + 16GB RAM (Layer 4)
- ✅ RTX 1650 Laptop + 16GB RAM (Layer 5)

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/immunex.git
cd immunex
```

### 2. Create Virtual Environment
```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Optional: GPU support
pip install torch==2.2.0+cu121 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### 4. Start Docker Services
```bash
docker compose up -d  # Starts Kafka, Elasticsearch, Redis, Ollama
docker ps  # Verify all services running
```

### 5. Download Pre-trained Models
Models are auto-downloaded on first run, or manually:
```bash
python layer1_detection/model.py --download-models
python layer2_correlation/train_bilstm.py --load-pretrained
```

---

## 🎯 Running IMMUNEX

### Single-Machine Setup (All Layers Local)
```bash
./start_immunex.sh
```
This script:
1. Kills previous processes
2. Starts Docker (Kafka, ES, Redis)
3. Starts all 5 layers sequentially
4. Starts Orchestrator on port 8000
5. Outputs health status

**Logs**: All layer logs in `./logs/`

### Distributed Setup (Multi-Machine)
```bash
# Machine 1 (your RTX 4070)
./start_immunex.sh

# Machine 2 (Acer w/ RTX 4050)
LAYER2_URL=http://10.0.0.2:8002 ./start_orchestrator.sh

# Configure in Orchestrator
export LAYER2_URL=http://10.0.0.2:8002
export LAYER3_URL=http://10.0.0.3:8003
export LAYER4_URL=http://10.0.0.4:8004
export LAYER5_URL=http://10.0.0.5:8005
```

### Stop All Services
```bash
./stop_immunex.sh
```

---

## 🔌 API Endpoints

### Orchestrator (Port 8000)

#### Health Check
```bash
GET /health
# Response: {status: "ok", layers: {1: true, 2: true, ...}}
```

#### Process Alert
```bash
POST /pipeline/run
Content-Type: application/json

{
  "alert_id": "alert-001",
  "source_ip": "192.168.1.100",
  "dest_ip": "10.0.0.50",
  "attack_type": "SQL_Injection",
  "timestamp": "2026-05-24T10:30:00Z",
  "severity": "high"
}

# Response:
{
  "pipeline_id": "uuid-xxx",
  "verdict": "ANOMALOUS",
  "l1_score": 0.85,
  "l2_stage": "Exploitation",
  "l3_action": "isolate",
  "l4_severity": "HIGH",
  "l5_next_predicted": "Exfiltration"
}
```

#### Demo Inject
```bash
POST /demo/inject
# Injects 5 pre-built test alerts, processes through full pipeline
```

#### Get Metrics
```bash
GET /metrics
# Returns: throughput, latency (p50/p95/p99), error rates
```

### Layer 1 (Detection)
```bash
POST /detect
{
  "log": "Raw syslog or JSON log",
  "source_hint": "syslog|json|csv"
}

POST /detect/batch
{
  "logs": ["log1", "log2", ...],
  "batch_size": 32
}

POST /ingest/batch
{
  "logs": [...],
  "run_full_pipeline": true
}
```

### Layer 2 (Correlation)
```bash
POST /correlate
{
  "alerts": [l1_result_1, l1_result_2, ...],
  "window_size": 50
}
```

### Layer 3 (Response)
```bash
POST /decide
{
  "attack_graph": {...},
  "feature_vector": [...]
}

POST /execute
{
  "action_index": 0,
  "dry_run": false
}
```

### Layer 4 (Immunity)
```bash
POST /classify
{
  "features": [77-dimensional array]
}

POST /retrain
{
  "dataset": "path/to/dataset.pt",
  "epochs": 10,
  "learning_rate": 0.001
}
```

### Layer 5 (Threat Memory)
```bash
GET /chains
# List all active attack chains

GET /chain/{chain_id}
# Full history of a specific chain

GET /attacker/{ip}
# Attacker profile by IP address

POST /chain/create
{
  "target_ip": "10.0.0.50",
  "first_observation": {...}
}
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# Orchestrator
export ORCHESTRATOR_TIMEOUT=30.0
export RATE_LIMIT_RPS=100
export RATE_LIMIT_BURST=200
export LOG_FORMAT=json  # json or standard

# Per-Layer Timeouts
export L1_TIMEOUT=15.0
export L2_TIMEOUT=20.0
export L3_TIMEOUT=30.0
export L4_TIMEOUT=15.0
export L5_TIMEOUT=20.0

# Layer URLs (for distributed setup)
export LAYER1_URL=http://localhost:8001
export LAYER2_URL=http://localhost:8002
export LAYER3_URL=http://localhost:8003
export LAYER4_URL=http://localhost:8004
export LAYER5_URL=http://localhost:8005

# Database
export ELASTICSEARCH_HOST=localhost:9200
export REDIS_HOST=localhost:6379
export KAFKA_BROKERS=localhost:9092

# GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Docker Compose
Edit `docker-compose.yml`:
```yaml
services:
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_BROKERS: kafka:9092
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  redis:
    image: redis:7-alpine
  ollama:
    image: ollama/ollama:latest
```

### Model Paths
```
layer1_detection/models/
├── faiss_index.bin
├── isolation_forest.pkl
└── layer1_scaler.pkl

layer2_correlation/models/
├── immunex_bilstm_phase3.pt
└── hmm_transition_counts.npy

layer4_immunity/models/
├── lora_model_ewc.pt
└── feature_scaler.pkl

Layer5_Threat Memory/
├── immunex_lstm_final.pt
└── immunex_hmm.pkl
```

---

## 🛠️ Development

### Project Structure
```
immunex/
├── orchestrator/          # Central coordination
│   ├── server.py         # FastAPI app
│   ├── ingest_api.py     # High-throughput ingest
│   └── security_status.py
├── layer1_detection/      # Anomaly detection
│   ├── server.py
│   ├── model.py
│   ├── inference.py
│   └── requirements.txt
├── layer2_correlation/    # Attack chain correlation
│   ├── server.py
│   ├── gat_model.py
│   ├── bilstm_model.py
│   └── hmm_predictor.py
├── Layer3 Response Engine/ # DQN-based response
│   └── Response_engine/
│       ├── main.py
│       ├── dqn_model.py
│       └── z3_verifier.py
├── layer4_immunity/       # Continual learning
│   ├── server.py
│   ├── train_77_features.py
│   └── lora_retrain.py
├── Layer5_Threat Memory/  # Temporal threat tracking
│   ├── server.py
│   ├── lstm_predictor.py
│   └── hmm_dwell_times.json
├── shared/               # Common utilities
│   ├── normalizer.py     # Log normalization
│   ├── schemas.py        # Pydantic models
│   ├── kafka_client.py
│   └── redis_client.py
├── docker-compose.yml
├── requirements.txt
└── start_immunex.sh
```

### Adding a New Feature

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Implement in appropriate layer**
   ```python
   # Example: Add new detection method in Layer 1
   # layer1_detection/my_detector.py
   def detect_anomaly(log):
       # Your implementation
       pass
   ```

3. **Add tests**
   ```bash
   python -m pytest tests/test_my_detector.py -v
   ```

4. **Commit & push**
   ```bash
   git add .
   git commit -m "feat: add my-feature to layer 1"
   git push origin feature/my-feature
   ```

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific layer
pytest tests/test_layer1.py -v

# With coverage
pytest tests/ --cov=orchestrator --cov=layer1_detection
```

### Training Custom Models

#### Layer 1: Train Isolation Forest
```bash
cd layer1_detection
python train_isolation_forest.py --dataset data/cicids2018.csv --epochs 50
```

#### Layer 2: Train BiLSTM
```bash
cd layer2_correlation
python train_bilstm.py --dataset data/l2_seq_dataset.pt --epochs 30 --batch-size 64
```

#### Layer 4: Fine-tune on Custom Data
```bash
cd layer4_immunity
python train_77_features.py \
  --dataset master_dataset/ \
  --lora-rank 8 \
  --epochs 20 \
  --learning-rate 0.001
```

---

## 🐛 Troubleshooting

### Layer Not Starting
```bash
# Check logs
tail -f logs/layer1.log

# Restart single layer
python -m uvicorn layer1_detection.server:app --reload --port 8001
```

### GPU Out of Memory
```bash
# Reduce batch size
export L1_BATCH_SIZE=16
export L2_BATCH_SIZE=8

# Or check GPU usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Kafka Connection Failed
```bash
# Check if running
docker ps | grep kafka

# Restart Docker services
docker compose restart kafka elasticsearch redis
```

### Model Loading Error
```bash
# Re-download models
rm layer*/models/*
python -c "import torch; torch.hub.load(...)"

# Or manually download from S3
aws s3 cp s3://immunex-models/weights/ . --recursive
```

### High Latency / Slow Inference
```bash
# Profile performance
python -m cProfile -s cumtime layer1_detection/inference.py

# Check if all GPU memory allocated
nvidia-smi

# Run in single-threaded mode if contested
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
```

---

## 📊 Performance Benchmarks

| Metric | Single GPU | 5-Layer Pipeline |
|--------|-----------|------------------|
| L1 Throughput | 10k logs/sec | 2-5k logs/sec |
| L1 Latency (p99) | 50ms | 150ms (total) |
| L2 Latency | 30ms/batch | 80ms |
| L3 Decision Time | 20ms | 50ms |
| Memory (all layers) | ~7GB | ~8GB |

**Note**: Distributed across GPU nodes achieves 40k+ logs/sec aggregate.

---

## 📝 License

Proprietary. All rights reserved by Barclays.

---

## 👥 Support & Contributing

For issues, security concerns, or feature requests:
1. Check existing issues: `https://github.com/yourusername/immunex/issues`
2. Contact the IMMUNEX team: `immunex-team@barclays.com`
3. Submit PRs following the contribution guidelines

---

## 🔐 Security Notes

- **Never commit credentials** – use `.env` (in `.gitignore`)
- **Model weights are binary** – handle carefully, don't expose paths
- **Production mode**: L3 has `DRY_RUN=false` (executes real actions)
- **HITL approval**: All high-severity actions require human review
- **Audit logging**: All decisions logged to Elasticsearch for compliance (RBI, GDPR, DORA)

---

**Last Updated**: May 24, 2026  
**Version**: 3.0.0