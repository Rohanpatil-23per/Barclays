# IMMUNEX — Multi-Layer Adaptive Threat Detection & Response System

![Version](https://img.shields.io/badge/version-3.0.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-orange)
![FastAPI](https://img.shields.io/badge/fastapi-0.135-green)
![License](https://img.shields.io/badge/license-Proprietary-red)

IMMUNEX is a five-layer AI-powered security pipeline built for financial institutions. It ingests raw network logs, detects anomalies, correlates attack chains, decides on a response action, learns from new threats, and builds persistent attacker profiles — all via independent FastAPI microservices coordinated by a central orchestrator.

Built for the **Barclays Hack-O-Hire** challenge.

---

## Table of Contents

- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running IMMUNEX](#running-immunex)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Training Custom Models](#training-custom-models)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

---

## Architecture

Each layer is an independent FastAPI service. The **Orchestrator** on port 8000 handles routing, circuit-breaking, rate limiting, and metrics. It fans each alert through all five layers and returns a unified verdict.

```
┌──────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR  :8000                          │
│        routing · circuit-breaking · metrics · rate-limiting       │
└──────────────────────────────────────────────────────────────────┘
         │              │              │              │
  ┌──────┴───┐   ┌──────┴───┐   ┌──────┴───┐   ┌──────┴───┐
  │ LAYER 1  │   │ LAYER 2  │   │ LAYER 3  │   │ LAYER 4  │
  │DETECTION │   │CORRELATION│  │ RESPONSE │   │ IMMUNITY │
  │  :8001   │   │  :8002   │   │  :8003   │   │  :8004   │
  │          │   │          │   │          │   │          │
  │ GATv2 +  │   │ BiLSTM + │   │ DQN +    │   │ LoRA +   │
  │AutoEncoder│  │Transformer│  │Z3 Prover │   │   EWC    │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                       │
                               ┌───────┴──────┐
                               │   LAYER 5    │
                               │THREAT MEMORY │
                               │    :8005     │
                               │ LSTM + HMM   │
                               │   + SQLite   │
                               └──────────────┘
```

**Docker services:** Kafka + Zookeeper, Elasticsearch, Redis, Kafka UI  
**Ollama** is started separately by `start_immunex.sh` (not in Docker Compose)

### Layer Details

**Layer 1 — Detection `:8001`**  
Identifies anomalous logs in real time using a GATv2 graph neural network and an AutoEncoder for novelty detection. Accepts raw syslog, JSON, or CSV. Outputs an anomaly score (0–1), confidence, embedding, and attack type classification.

**Layer 2 — Correlation `:8002`**  
Links related alerts into attack chains using a Transformer spatial encoder, BiLSTM narrative tracker, and HMM predictor. Maps chains to MITRE ATT&CK stages and predicts the next likely stage. Operates over 50-alert sequence windows.

**Layer 3 — Response Engine `:8003`**  
Selects a remediation action (`do_nothing`, `monitor`, `block`, `isolate`) using a Dueling DQN policy. Every proposed action is formally verified by a Z3 theorem prover before execution. High-severity actions require human approval before going live.

**Layer 4 — Immunity `:8004`**  
Classifies threat severity (Benign / Low / Medium / High / Critical) using a Logistic Regression model with LoRA adapters and EWC (Elastic Weight Consolidation) for continual learning without catastrophic forgetting. Trained on a 77-feature CICIDS 2018 representation — 95.88% accuracy on the held-out test set.

**Layer 5 — Threat Memory `:8005`**  
Builds persistent attacker profiles using an LSTM + HMM combo backed by SQLite. Tracks attack sequences across sessions and predicts future moves per attacker IP.

---

## System Requirements

| Component | Minimum | Notes |
|---|---|---|
| Python | 3.9+ | — |
| RAM | 16 GB | 32 GB recommended for all layers on one machine |
| GPU | NVIDIA CUDA 12.6+ | CPU fallback works but is slow |
| Disk | ~15 GB | pip deps + model weights + Docker volumes |
| OS | Linux / macOS | Windows needs WSL2 |
| Docker | any recent version | For Kafka, Elasticsearch, Redis |

> **Windows note:** The startup script uses `python.exe`. On Linux/macOS, edit `start_immunex.sh` and replace `python.exe` with `python3`.

### Tested distributed node setup

The repo ships WireGuard config files for the five machines used during development:

| Config file | GPU | RAM | Role |
|---|---|---|---|
| `peer1_fedora.conf` | RTX 4070 | 32 GB | Layer 1 + Orchestrator |
| `peer2_acer.conf` | RTX 4050 | 16 GB | Layer 2 |
| `peer3_lenovo.conf` | RTX 3050 | 24 GB | Layer 3 |
| `peer4_victus.conf` | RTX 2050 | 16 GB | Layer 4 |
| `peer5_pavilion.conf` | RTX 1650 | 12 GB | Layer 5 |

---

## Installation

```bash
git clone https://github.com/Rohanpatil-23per/Barclays.git
cd Barclays

# The startup script expects the venv here
python3 -m venv ~/.venvs/immunex
source ~/.venvs/immunex/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Start Docker services (Kafka, Elasticsearch, Redis):

```bash
docker compose up -d
docker ps   # verify all containers are running
```

---

## Running IMMUNEX

### Single machine

```bash
./start_immunex.sh
```

The script will:
1. Kill any processes on ports 8000–8005
2. Wipe and restart Docker services
3. Start Ollama (skips if already running)
4. Start all five layers one by one, waiting for each `/health` endpoint
5. Start the Orchestrator on port 8000
6. Print a live health summary

```bash
./stop_immunex.sh    # stop everything
./check_mesh.sh      # verify WireGuard mesh connectivity
```

Logs for each layer are written to `./logs/`.

### Distributed mode

Set the remote URLs before running the startup script. Any layer whose URL points to `localhost` is started locally; the rest are assumed to already be running on their respective machines.

```bash
LAYER2_URL=http://10.0.0.2:8002 \
LAYER3_URL=http://10.0.0.3:8003 \
LAYER4_URL=http://10.0.0.4:8004 \
LAYER5_URL=http://10.0.0.5:8005 \
./start_immunex.sh
```

To start a single layer manually on a remote node:

```bash
source ~/.venvs/immunex/bin/activate
uvicorn layer2_correlation.server:app --host 0.0.0.0 --port 8002
```

---

## API Reference

### Orchestrator `:8000`

#### Health check
```http
GET /health
```
```json
{ "status": "ok", "layers": { "1": true, "2": true, "3": true, "4": true, "5": true } }
```

#### Run an alert through the full pipeline
```http
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
```
```json
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

#### Inject demo alerts
```http
POST /demo/inject
```
Runs five pre-built test scenarios through the full pipeline.

#### Metrics
```http
GET /metrics
```
Returns throughput, latency percentiles (p50/p95/p99), and per-layer error rates.

---

### Layer 1 — Detection `:8001`

```http
POST /detect
{ "log": "<raw syslog or JSON>", "source_hint": "syslog|json|csv" }

POST /detect/batch
{ "logs": ["log1", "log2", ...], "batch_size": 32 }

POST /ingest/batch
{ "logs": [...], "run_full_pipeline": true }
```

### Layer 2 — Correlation `:8002`

```http
POST /correlate
{ "alerts": [<l1_result>, ...], "window_size": 50 }
```

### Layer 3 — Response Engine `:8003`

```http
POST /decide
{ "attack_graph": {}, "feature_vector": [...] }

POST /execute
{ "action_index": 0, "dry_run": true }
```

### Layer 4 — Immunity `:8004`

```http
POST /classify
{ "features": [<77-dimensional CICIDS feature vector>] }

POST /retrain
{ "dataset": "path/to/dataset.pt", "epochs": 10, "learning_rate": 0.001 }
```

### Layer 5 — Threat Memory `:8005`

```http
GET  /chains                 # list all active attack chains
GET  /chain/{chain_id}       # full history for a chain
GET  /attacker/{ip}          # attacker profile by IP
POST /chain/create
{ "target_ip": "10.0.0.50", "first_observation": {} }
```

---

## Environment Variables

```bash
# Orchestrator
ORCHESTRATOR_TIMEOUT=30.0
RATE_LIMIT_RPS=100
RATE_LIMIT_BURST=200
LOG_FORMAT=json                  # json | standard

# Per-layer timeouts
L1_TIMEOUT=15.0
L2_TIMEOUT=20.0
L3_TIMEOUT=30.0
L4_TIMEOUT=15.0
L5_TIMEOUT=20.0

# Layer URLs — set these to enable distributed mode
LAYER1_URL=http://localhost:8001
LAYER2_URL=http://localhost:8002
LAYER3_URL=http://localhost:8003
LAYER4_URL=http://localhost:8004
LAYER5_URL=http://localhost:8005

# Infrastructure
ELASTICSEARCH_HOST=localhost:9200
REDIS_HOST=localhost:6379
KAFKA_BROKERS=localhost:9092

# GPU
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Store secrets in a `.env` file — it's already in `.gitignore`.

---

## Project Structure

```
Barclays/
├── orchestrator/               # Central coordinator (FastAPI)
│   ├── server.py
│   ├── ingest_api.py
│   └── security_status.py
├── layer1_detection/           # Anomaly detection
│   ├── server.py
│   ├── model.py                # GATv2 + AutoEncoder
│   └── inference.py
├── layer2_correlation/         # Attack chain correlation
│   ├── server.py
│   ├── gat_model.py
│   ├── bilstm_model.py
│   └── hmm_predictor.py
├── Layer3 Response Engine/     # DQN response + Z3 verification
│   └── Response_engine/
│       ├── main.py
│       ├── dqn_model.py
│       └── z3_verifier.py
├── layer4_immunity/            # Continual learning (LoRA + EWC)
│   ├── server.py
│   ├── train_77_features.py
│   └── lora_retrain.py
├── Layer5_Threat Memory/       # Temporal attacker tracking
│   ├── server.py
│   ├── lstm_predictor.py
│   └── hmm_dwell_times.json
├── shared/                     # Common utilities
│   ├── normalizer.py
│   ├── schemas.py
│   ├── kafka_client.py
│   └── redis_client.py
├── CICIDS2018/                 # Training dataset
├── certs/                      # mTLS certificates (Layer 5)
├── logs/                       # Runtime logs (gitignored)
├── tests/
├── docker-compose.yml
├── requirements.txt
├── start_immunex.sh
├── stop_immunex.sh
└── check_mesh.sh
```

---

## Training Custom Models

### Layer 1 — Isolation Forest
```bash
cd layer1_detection
python train_isolation_forest.py --dataset data/cicids2018.csv --epochs 50
```

### Layer 2 — BiLSTM
```bash
cd layer2_correlation
python train_bilstm.py --dataset data/l2_seq_dataset.pt --epochs 30 --batch-size 64
```

### Layer 4 — LoRA fine-tune on custom data
```bash
cd layer4_immunity
python train_77_features.py \
  --dataset master_dataset/ \
  --lora-rank 8 \
  --epochs 20 \
  --learning-rate 0.001
```

---

## Troubleshooting

**A layer isn't starting**
```bash
tail -f logs/layer1.log

# Restart a single layer manually
uvicorn layer1_detection.server:app --reload --port 8001
```

**CUDA out of memory**
```bash
export L1_BATCH_SIZE=16
export L2_BATCH_SIZE=8
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Kafka connection refused**
```bash
docker ps | grep kafka
docker compose restart kafka zookeeper
```

**Elasticsearch not ready**
```bash
curl http://localhost:9200/_cluster/health
docker compose restart elasticsearch
```

**High latency**
```bash
nvidia-smi                               # check GPU utilisation
python -m cProfile -s cumtime layer1_detection/inference.py
export CUDA_LAUNCH_BLOCKING=1            # if memory is being contested
```

---

## Security Notes

- **No hardcoded secrets** — use a `.env` file (already in `.gitignore`)
- **Layer 3 dry-run** — `dry_run=true` by default; set to `false` in production only — it executes real network actions
- **Human-in-the-loop** — high-severity `block` and `isolate` actions require manual approval before execution
- **mTLS on Layer 5** — inter-node traffic uses mutual TLS; certs live in `certs/`. Single-machine mode skips the TLS flags automatically
- **Audit logging** — all pipeline decisions are persisted to Elasticsearch

---

## Dataset

Layer 4 is trained and evaluated on the **CICIDS 2018** intrusion detection dataset with a 77-feature representation. Raw CSVs and pre-processed tensors are under `CICIDS2018/`.

---

*Barclays Hack-O-Hire 2026 — Team PuranPoli Enjoyers*
