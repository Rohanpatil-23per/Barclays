# IMMUNEX Layer 3 — Immune Response Engine

This folder contains the core reinforcement learning models (e.g., `dueling_dqn_immunex.zip`) used by Layer 3 to autonomously react to correlated cyber threats from Layer 2.

---

## 🏗️ Architecture & Progress Summary

We have successfully built the complete end-to-end inference and execution pipeline for **Layer 3** of the IMMUNEX system. It is designed to run in a fully air-gapped, offline banking environment.

### 1. **DQN Inference Engine** (`response_engine.py`)
- Loads the Stable-Baselines3 Dueling DQN model.
- Accepts a 128-dimensional correlated feature vector representing a network anomaly.
- Maps the state to one of 50 discrete containment, investigation, or monitoring actions while exposing the raw Q-values for interpretability.

### 2. **Z3 Formal Safety Verifier** (`safety_verifier.py`)
- Before any action executes, it is mathematically verified against strict RBI, GDPR, and DORA banking compliance rules.
- Contains constraints for:
  - **Trading Window Protection**: Prevents aggressive network blocks during core banking hours (09:00-17:00 IST).
  - **Severity Floors**: Ensures isolation actions only occur for high/critical severities.
  - **Self-Harm Protection**: Hardcoded IP whitelist protecting the IMMUNEX management console from being blackholed.
  - **Rollback Cascade**: Ensures backups exist before allowing rollback operations.

### 3. **Human-in-the-Loop & LLM Reasoning** (`llm_reasoning.py`)
- Routes AI response decisions to a locally hosted **Llama-3** (via Ollama).
- The LLM assesses the business risk of the AI's intended action. 
- If the risk is high, it holds the pipeline and drops into a **Human-in-the-loop (HITL)** approval gate, allowing a SOC analyst to accept or reject the action (auto-substituting a safe parallel action if rejected).

### 4. **Action Execution Harness** (`action_executor.py`)
- A modular dispatch table capable of executing all 50 mitigation actions.
- Currently operates in `dry_run` prototype mode to log output structures and timings without running destructive system calls yet.

### 5. **Playbook Generator** (`playbook_generator.py`)
- Post-execution, the local Llama-3 model dynamically generates an incident response report explaining *why* the attack happened and *how* the system fought it.
- Has a 100% offline rule-based fallback system that guarantees a playbook is generated even if the LLM crashes or times out.

### 6. **FastAPI Inference Service** (`main.py`)
- Wraps all the modules above into a high-performance REST API.
- Endpoints:
  - `POST /respond`: The main entrypoint for Layer 2 alerts.
  - `POST /approve/{alert_id}`: Human approval gateway for intercepted actions in the pending queue.
  - `GET /health` & `GET /actions`: System diagnostics.

### 7. **End-to-End Validation**
- Developed `test_e2e.py` which rigorously validates:
  - 10-request highly concurrent throughput (avg < 50ms without LLM).
  - Accurate HITL terminal interception.
  - Simulated Llama-3 application failures (proving the fallback works flawlessly without cascading failures).
