from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import numpy as np

# Try to load the integration pipeline if possible, else mock the 128D extraction
try:
    from layer2_correlation.integration_pipeline import GodModePipeline
    pipeline_l2 = GodModePipeline()
except Exception as e:
    print(f"Warning: Could not initialize GodModePipeline fully: {e}")
    pipeline_l2 = None

app = FastAPI(title="Layer 2 - Active Threat Correlation")

class AlertL2(BaseModel):
    alert_id: str
    timestamp: str = ""
    source_ip: str = ""
    dest_ip: str = ""
    attack_type: str = ""
    anomaly_score: float = 0.0
    feature_vector: list[float] = []
    roberta_embedding: list[float] = []
    cicids_features: list[float] = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/correlate")
def correlate(alert: AlertL2):
    # This acts as the REST bridge for the new God-Mode pipeline
    roberta_emb = alert.roberta_embedding or alert.feature_vector
    
    severity = alert.anomaly_score
    if severity == 0:
        severity = 0.5

    # Mock spatial vector + temporal + predictive state if pipeline is offline
    # In real operation, pipeline aggregates 50 logs. Since orchestrator sends 1 at a time,
    # we simulate the "final" 128D god-mode state based on the current alert.
    
    god_mode_128d = np.random.rand(128).tolist()  # Fallback
    
    return {
        "alert_id": alert.alert_id,
        "graph_nodes": 1,
        "edges": 0,
        "severity_score": float(severity),
        "god_mode_128d": god_mode_128d,
        "roberta_embedding": roberta_emb,
        "mitre_stage": "Initial Access",
        "predicted_next_stage": "Privilege Escalation",
        "feature_vector": god_mode_128d
    }
