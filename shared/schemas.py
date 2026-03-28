from pydantic import BaseModel, ConfigDict, model_validator
from typing import List, Optional, Any

class Alert(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    alert_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    alert_type: Optional[str] = "network"
    severity: Optional[str] = "medium"
    features: Optional[List[Optional[float]]] = None
    text: Optional[str] = None
    event_type: Optional[str] = None
    protocol: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    process: Optional[str] = None
    file: Optional[str] = None
    privilege_level: Optional[str] = None

class AnomalyResult(BaseModel):
    alert_id: str
    timestamp: Optional[str]
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    attack_type: Optional[str] = "unknown"
    anomaly_score: float
    is_anomalous: bool
    embedding: List[float]
    detection_method: Optional[str] = "unknown"
    event_type: Optional[str] = None
    protocol: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    process: Optional[str] = None
    file: Optional[str] = None
    privilege_level: Optional[str] = None
    confidence: float
    cicids_features: Optional[List[float]] = None

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
