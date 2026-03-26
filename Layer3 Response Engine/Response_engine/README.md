

# 🛡️ IMMUNEX Layer 3 — Immune Response Engine

## 🚀 Overview
IMMUNEX Layer 3 is an **autonomous, priority-driven, and formally verified cyber incident response system** designed for Zero-Trust banking environments.

It ingests 128-dimensional threat vectors from Layer 2, determines the optimal multi-step containment strategy using Deep Reinforcement Learning (DQN), and physically verifies the safety of those actions using a Z3 SMT Solver before executing them. It strictly enforces **human-in-the-loop approval** for all high-impact actions, utilizing PostgreSQL and `pgvector` to learn from human overrides via Reinforcement Learning from Human Feedback (RLHF).

---

## 🧠 Key Features
* 🔹 **Multi-Action Dueling DQN:** Analyzes complex threat states to select from 50 discrete containment actions.
* 🔹 **Z3 Formal Safety Verification:** The "Math Cop" that proves mathematically that AI actions won't violate strict banking rules (e.g., blocking network isolation during market trading hours).
* 🔹 **Zero-Trust Human-in-the-Loop:** Absolutely NO auto-execution for critical alerts. Administrators can *Approve*, *Reject*, or *Override* the AI.
* 🔹 **RLHF Memory (`pgvector`):** The system stores human rejections as negative signals and human overrides as positive signals in PostgreSQL for continuous offline model retraining.
* 🔹 **Local LLM Playbook Generation:** Uses Ollama (LLaMA-3) to dynamically write human-readable SOC rationales and compliance-mapped playbooks.
* 🔹 **Immutable Audit Logging:** Atomic JSON Lines writing that captures exact execution provenance (`auto_approved`, `human_reviewed`, `approver_id`) for RBI/GDPR/DORA compliance.

---

## 🏗️ System Architecture

```text
Layer 2 Alert (128-dim state vector)
    │
    ▼
DQN Inference ──(Cosine similarity check against rejected_demonstrations)
    │
    ▼
Conflict Resolution (Removes mutually exclusive actions)
    │
    ▼
Z3 Safety Verification (Pass 1)
    │
    ▼
LLM Reasoning (Risk & Explanation)
    │
    ▼
Priority Queue (Heap + Lookup)
    │
    ▼
Human Decision Gate (Mandatory)
   ╱        │        ╲
REJECT   APPROVE   OVERRIDE (Human selects new actions)
  │         │         │
  │         │         ▼
  │         │      Z3 Verification (Pass 2 - Validates Human)
  │         │         │
  │         ▼         ▼
  │      Validated Execution
  │         │         │
  ▼         ▼         ▼
pgvector RLHF Database (expert_demonstrations & rejected_demonstrations)
            │
            ▼
     Immutable Audit Logging
```

---

## ⚙️ Tech Stack
* **Core Backend:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning:** Stable-Baselines3 (DQN), PyTorch
* **Formal Verification:** Z3 Theorem Prover
* **Database & RLHF:** PostgreSQL, `asyncpg`, `pgvector`
* **Local LLM:** Ollama (LLaMA-3 8B)
* **Observability:** Structlog (JSON formatted)

---

## 📂 Project Structure
```text
response_engine/
├── main.py                # FastAPI entry point & API routes
├── response_engine.py     # DQN inference & conflict filtering
├── safety_verifier.py     # Z3 constraint verification logic
├── database.py            # PostgreSQL + pgvector connection pool
├── action_executor.py     # Execution stubs & pre/post validation
├── playbook_generator.py  # LLM incident report generation
├── audit_logger.py        # Immutable JSONL compliance logging
├── action_registry.py     # Global dictionary of 50 actions
└── llm_reasoning.py       # LLM risk classification layer
```

---

## ▶️ How to Run

### 1. Start the Database (Docker)
Ensure Docker is running, then spin up the PostgreSQL + `pgvector` instance:
```bash
cd response_engine
docker compose up -d
```

### 2. Install Dependencies
```bash
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
```

### 3. Start the Server
```bash
# Point the app to the local Docker database
$env:DATABASE_URL="postgresql://postgres:secret@localhost:5433/immunex"

# Run FastAPI
uvicorn response_engine.main:app --port 8001 --reload
```

---

## 🔌 API Endpoints

### 🔹 Run Pipeline
```http
POST /respond
```
*Ingests the 128-dim Layer 2 alert, runs inference, verifies safety, and parks the alert in the priority queue.*

### 🔹 View Pending Approvals
```http
GET /pending
```

### 🔹 Human Decision Endpoints
```http
POST /approve/{alert_id}    # Executes DQN's suggested actions
POST /admin/reject          # Drops alert & stores as negative RLHF signal
POST /admin/override        # Executes human's chosen actions & stores as positive RLHF signal
```

### 🔹 System Health & Information
```http
GET /health                 # Checks DQN, Database, and Ollama status
GET /actions                # Lists all 50 available containment actions
```

---

## 📊 Priority System
Alerts waiting in the queue are sorted by priority to ensure critical threats are surfaced first.

| Severity | Base Priority |
| -------- | -------- |
| Critical | 1        |
| High     | 2        |
| Medium   | 3        |
| Low      | 4        |

* **Boost Conditions:** If the DQN calculates a "High Impact" response or registers "Low Confidence" (<60%), the priority is dynamically boosted by 1 level.

---

## 🔐 Zero-Trust Human-in-the-Loop
* ✅ **No Auto-Execution:** The auto-execute bypass has been permanently removed.
* ✅ **Z3 Double-Pass:** If a human overrides the AI, the human's choices must *also* pass Z3 mathematical safety verification before execution.
* ✅ **Race-Condition Safe:** Queue popping is thread-safe to prevent 500 errors if multiple admins approve an alert simultaneously.

---

## 🧾 Audit Logging
Logs are stored atomically to prevent corruption:
```text
audit_logs/audit_YYYY-MM-DD.jsonl
```
Each JSON Lines entry includes:
* 128-dim state vector & Alert payload
* DQN Actions vs. Human Executed Actions
* Z3 Verification status & reasons
* **Provenance Tracking:** `auto_approved`, `human_reviewed`, and `approver_id`.

---

## ⚠️ Important Notes
* Ensure your `models/dueling_dqn_immunex.zip` file is placed in the project root.
* The system defaults to **dry-run mode** (simulated execution).
* If Ollama or PostgreSQL are unreachable, the system **degrades gracefully** (using rule-based fallback playbooks and skipping RLHF writes) to ensure the API never stalls.

---

## 🚀 Future Improvements
* Complete the offline Python script to pull `expert_demonstrations` and `rejected_demonstrations` from Postgres to automatically retrain the DQN weights.
* Integrate an external Redis cluster for distributed queue scaling.
* Connect the FastAPI endpoints to a React/Next.js frontend dashboard for SOC analysts.

---

