from pydantic import BaseModel
from typing import List, Optional, Any

class Alert(BaseModel):
    alert_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_ip: str
    dest_ip: Optional[str] = None
    alert_type: Optional[str] = "network"
    severity: Optional[str] = "medium"
    features: Optional[List[float]] = None
    text: Optional[str] = None

class AnomalyResult(BaseModel):
    alert_id: str
    timestamp: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    attack_type: Optional[str] = "unknown"
    anomaly_score: float
    is_anomalous: bool
    embedding: List[float]
    detection_method: Optional[str] = "unknown"
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
