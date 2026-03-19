from pydantic import BaseModel
from typing import List, Optional

class Alert(BaseModel):
    alert_id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    alert_type: str
    severity: float
    features: List[float]

class AnomalyResult(BaseModel):
    alert_id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    attack_type: str
    anomaly_score: float
    is_anomalous: bool
    embedding: List[float]
    detection_method: str
    confidence: float

class AttackGraph(BaseModel):
    chain_id: str
    nodes: List[dict]
    edges: List[dict]
    predicted_next_stage: str
    confidence: float

class ResponseAction(BaseModel):
    chain_id: str
    action: str
    target_ip: str
    verified_safe: bool
    q_value: float

class Playbook(BaseModel):
    incident_id: str
    attack_summary: str
    steps: List[str]
    predicted_next: str
    confidence: float
