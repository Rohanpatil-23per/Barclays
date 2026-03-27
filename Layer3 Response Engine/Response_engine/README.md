# IMMUNEX Layer 3 — Immune Response Engine

## 🚀 Overview

IMMUNEX Layer 3 is a **priority-driven, human-in-the-loop cyber incident response system** designed for banking-grade environments.

It processes alerts from Layer 2, applies AI-driven decision making, ensures safety via formal verification, and enforces **strict human approval before any action execution**.

---

## 🧠 Key Features

* 🔹 **Multi-action DQN (Reinforcement Learning)**
* 🔹 **Z3 Safety Verification**
* 🔹 **LLM-based Reasoning (Risk + Explanation)**
* 🔹 **Priority Queue (Critical-first handling)**
* 🔹 **Human-in-the-loop (ALL actions require approval)**
* 🔹 **Playbook Generation (LLM)**
* 🔹 **Execution Validation Layer**
* 🔹 **Immutable Audit Logging (JSONL, compliance-ready)**

---

## 🏗️ System Architecture

```
Layer 2 Alert
    ↓
DQN → multi-action output
    ↓
Conflict Resolution
    ↓
Z3 Verification
    ↓
LLM Reasoning
    ↓
Priority Calculation (1–4)
    ↓
Queue (heap + lookup)
    ↓
Human Approval (mandatory)
    ↓
Playbook Override
    ↓
Validated Execution
    ↓
Audit Logging
```

---

## ⚙️ Tech Stack

* **Python (FastAPI)**
* **Stable-Baselines3 (DQN)**
* **Z3 Solver**
* **Ollama (LLaMA 3)**
* **Structlog (logging)**
* **JSONL Audit Logs**

---

## 📂 Project Structure

```
response_engine/
├── main.py                 # FastAPI entry point
├── response_engine.py     # DQN inference logic
├── safety_verifier.py     # Z3 constraint verification
├── action_executor.py     # Execution + validation
├── playbook_generator.py  # LLM playbook generation
├── audit_logger.py        # Immutable audit logs
├── action_registry.py     # Action definitions
├── llm_reasoning.py       # LLM reasoning layer
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start server

```bash
uvicorn response_engine.main:app --port 8001 --reload
```

---

## 🔌 API Endpoints

### 🔹 Run Pipeline

```http
POST /respond
```

---

### 🔹 View Pending Approvals

```http
GET /pending
```

---

### 🔹 Approve / Reject / Modify

```http
POST /approve/{alert_id}
```

Example:

```json
{
  "approved": true,
  "modified_actions": [10, 12]
}
```

---

### 🔹 Health Check

```http
GET /health
```

---

### 🔹 List Actions

```http
GET /actions
```

---

## 🧪 Testing Workflow

### Step 1: Send alert

```bash
POST /respond
```

### Step 2: Check queue

```bash
GET /pending
```

### Step 3: Approve / Reject

```bash
POST /approve/{alert_id}
```

---

## 📊 Priority System

| Severity | Priority |
| -------- | -------- |
| Critical | 1        |
| High     | 2        |
| Medium   | 3        |
| Low      | 4        |

### Boost Conditions:

* High impact
* Low confidence (uncertain)

---

## 🔐 Human-in-the-Loop

* ✅ Every action requires approval
* ❌ No auto-execution
* ✅ Supports modification before execution

---

## 🧾 Audit Logging

Logs stored in:

```
audit_logs/audit_YYYY-MM-DD.jsonl
```

Each entry includes:

* Alert details
* Actions (before/after)
* Priority & uncertainty
* Approval status
* Execution result
* RL feedback (state, action, outcome)

---

## ⚠️ Important Notes

* Model file is not included (add manually)
* System runs in **dry-run mode by default**
* Rate limiter is currently a **mock implementation**

---

## 🚀 Future Improvements

* Redis-based distributed queue
* UI dashboard for approvals
* Real-time monitoring
* RL retraining pipeline from logs
* Advanced parameter validation per action

---

## 👨‍💻 Author

IMMUNEX Layer 3 — Cyber Defense AI System
Built as a **production-grade intelligent response engine**

---

## ⭐ Final Note

This project demonstrates a **real-world AI-driven security system**, combining:

* Reinforcement Learning
* Formal Verification
* Human Oversight
* Explainable AI

👉 Designed for **banking-grade cybersecurity environments**
