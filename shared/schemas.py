# shared/schemas.py
# IMMUNEX — Inter-layer Data Contracts
# ──────────────────────────────────────────────────────────────
# FROZEN after team agreement on Day 1.
# ANY change requires group approval — message Person 5.
# Owner: Person 5
# ──────────────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import List, Optional


# ── RAW INPUT ────────────────────────────────────────────────
# Source: SIEM / EDR / network tap
# Consumer: Layer 1 /detect endpoint
# Features: 77 normalized floats from CICFlowMeter
# (Protocol, Flow Duration, Total Fwd Packets, ... Idle Min)

class Alert(BaseModel):
    alert_id: str               # unique ID e.g. "alert_20260322_001"
    timestamp: str              # ISO format "2026-03-22T10:00:00"
    source_ip: str              # attacker IP
    dest_ip: str                # target IP
    alert_type: str             # raw label from sensor e.g. "port_scan"
    severity: float             # 0.0 to 1.0 from sensor
    features: List[float]       # exactly 77 floats, StandardScaler normalized
                                # order matches feature_columns.json


# ── LAYER 1 OUTPUT → LAYER 2 + LAYER 4 INPUT ────────────────
# Producer: Person 1 (RTX 4070) — RoBERTa + Isolation Forest
# Consumer: Person 2 /correlate endpoint
#           Person 4 /immunize endpoint (blind spot testing)

class AnomalyResult(BaseModel):
    alert_id: str               # echoed from Alert
    anomaly_score: float        # 0.0 (normal) to 1.0 (highly anomalous)
    is_anomalous: bool          # True if anomaly_score > threshold (0.5)
    attack_type: str            # predicted class e.g. "DoS", "Botnet", "Benign"
    confidence: float           # model confidence 0.0 to 1.0
    embedding: List[float]      # 768-dim CLS token from RoBERTa (for FAISS)
    detection_method: str       # "transformer" | "isolation_forest" | "ensemble"
    source_ip: str              # echoed from Alert for Layer 2 graph building
    dest_ip: str                # echoed from Alert for Layer 2 graph building
    timestamp: str              # echoed from Alert for Layer 2 temporal encoding


# ── LAYER 2 OUTPUT → LAYER 3 INPUT ──────────────────────────
# Producer: Person 2 (RTX 4050) — GATv2 + BiLSTM
# Consumer: Person 3 /respond endpoint

class AttackNode(BaseModel):
    node_id: str                # unique node identifier
    ip: str                     # IP address of this node
    attack_type: str            # MITRE ATT&CK technique e.g. "T1046"
    tactic: str                 # MITRE tactic e.g. "Reconnaissance"
    timestamp: str              # when this stage was observed
    severity: float             # node-level severity score

class AttackEdge(BaseModel):
    source_node_id: str         # from node
    target_node_id: str         # to node
    confidence: float           # GATv2 attention weight for this edge
    time_delta_seconds: float   # time between source and target events

class AttackGraph(BaseModel):
    chain_id: str               # unique chain identifier
    nodes: List[AttackNode]
    edges: List[AttackEdge]
    predicted_next_tactic: str  # next predicted MITRE tactic
    predicted_next_technique: str  # next predicted MITRE technique
    chain_confidence: float     # overall chain reconstruction confidence
    attack_stage: str           # current kill chain stage
    asset_criticality: float    # 0.0 to 1.0, how critical is the target


# ── LAYER 3 OUTPUT → LAYER 5 INPUT ──────────────────────────
# Producer: Person 3 (RTX 3050) — Dueling DQN + Z3 solver
# Consumer: Person 5 /explain endpoint

class ResponseAction(BaseModel):
    chain_id: str               # echoed from AttackGraph
    action: str                 # selected response e.g. "isolate_host"
    action_description: str     # human readable e.g. "Block IP 203.0.113.5"
    target_ip: str              # IP to act on
    target_asset: str           # asset name e.g. "web-server-01"
    verified_safe: bool         # True = Z3 solver confirmed no constraint violation
    q_value: float              # DQN Q-value for selected action
    z3_constraints_checked: int # number of safety constraints evaluated
    business_hours: bool        # True if action taken during market hours
    requires_human_approval: bool  # True for high-impact actions


# ── LAYER 4 OUTPUT (standalone, no downstream consumer) ──────
# Producer: Person 4 (RTX 2050) — LoRA retraining + AIS mutations
# Consumer: Dashboard only (for live metrics display)

class ImmunityStatus(BaseModel):
    cycle_id: int               # retraining cycle number
    blind_spots_found: int      # number of evasions detected this cycle
    mutations_tested: int       # total attack variants tested
    retrained: bool             # True if retraining was triggered
    accuracy_before: float      # detection accuracy before retraining
    accuracy_after: float       # detection accuracy after retraining
    lora_loss: float            # LoRA fine-tuning loss
    ewc_penalty: float          # EWC regularization penalty


# ── LAYER 5 OUTPUT → DASHBOARD ───────────────────────────────
# Producer: Person 5 (GTX 1650) — LSTM-HMM + Phi-3-mini + Neo4j
# Consumer: Streamlit dashboard

class ThreatPrediction(BaseModel):
    current_stage: str          # current MITRE tactic
    next_stage: str             # predicted next tactic
    time_to_next_hours: float   # estimated hours until next stage
    confidence: float           # HMM prediction confidence

class Playbook(BaseModel):
    incident_id: str            # unique incident identifier
    chain_id: str               # echoed from AttackGraph
    attack_summary: str         # 2-3 sentence LLM summary
    steps: List[str]            # ordered response steps from LLM
    threat_actor_match: str     # closest MITRE ATT&CK group e.g. "APT28"
    threat_prediction: ThreatPrediction
    neo4j_similar_incidents: int   # number of similar past incidents found
    overall_severity: str       # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommended_escalation: bool   # True if human SOC team should be paged


# ── PIPELINE FLOW SUMMARY ────────────────────────────────────
#
#  Alert
#    → [Layer 1] → AnomalyResult
#                      → [Layer 2] → AttackGraph
#                      |                → [Layer 3] → ResponseAction
#                      |                                  → [Layer 5] → Playbook
#                      |                                                    → Dashboard
#                      → [Layer 4] → ImmunityStatus
#                                        → Dashboard
#
# All layers expose:
#   POST /<action>  → input/output schemas above
#   GET  /health    → {"status": "ok", "model": "loaded", "layer": N}
#
# Port assignments:
#   Layer 1 → 8001    Layer 2 → 8002    Layer 3 → 8003
#   Layer 4 → 8004    Layer 5 → 8005
#
# ──────────────────────────────────────────────────────────────