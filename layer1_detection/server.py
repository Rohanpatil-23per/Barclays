import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from shared.schemas import Alert, AnomalyResult
from layer1_detection.inference_utils import load_top_features, serialize_features

ROBERTA_PATH   = "models/roberta_layer1"
ISO_PATH       = "models/isolation_forest.pkl"
SCALER_PATH    = "models/layer1_scaler.pkl"
STATS_PATH     = "models/iso_forest_stats.json"
TOP_FEATS_PATH = "master_dataset/top_features.json"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN        = 128

app = FastAPI(title="IMMUNEX Layer 1 — Innate Detection", version="1.0")

tokenizer     = None
roberta       = None
iso_forest    = None
scaler        = None
top_features  = None
iso_threshold = None

@app.on_event("startup")
async def load_models():
    global tokenizer, roberta, iso_forest, scaler, top_features, iso_threshold
    print(f"Loading models on {DEVICE}...")

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_PATH)
    roberta   = RobertaForSequenceClassification.from_pretrained(ROBERTA_PATH)
    roberta.to(DEVICE)
    roberta.eval()

    with open(ISO_PATH, "rb") as f:
        iso_forest = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    top_features = load_top_features(TOP_FEATS_PATH)

    with open(STATS_PATH) as f:
        iso_threshold = json.load(f)["threshold"]

    print("All models loaded ✅")

@app.get("/health")
def health():
    return {"status": "ok", "layer": 1, "device": str(DEVICE)}

@app.post("/detect", response_model=AnomalyResult)
async def detect(alert: Alert):
    if len(alert.features) != 77:
        raise HTTPException(400, f"Expected 77 features, got {len(alert.features)}")

    features  = np.array(alert.features).reshape(1, -1)
    scaled    = scaler.transform(features)
    iso_score = float(iso_forest.decision_function(scaled)[0])
    iso_flag  = iso_score < iso_threshold

    text = serialize_features(alert.features, top_features)
    enc  = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                     truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs     = roberta(
            input_ids      = enc['input_ids'].to(DEVICE),
            attention_mask = enc['attention_mask'].to(DEVICE)
        )
        attack_prob = float(torch.softmax(outputs.logits, dim=1)[0][1].cpu())
        embedding   = roberta.roberta(
            input_ids      = enc['input_ids'].to(DEVICE),
            attention_mask = enc['attention_mask'].to(DEVICE)
        ).last_hidden_state[:, 0, :].squeeze().cpu().tolist()

    is_anomalous  = attack_prob > 0.5 or iso_flag
    anomaly_score = float(np.clip(max(attack_prob, float(iso_flag) * 0.8), 0.0, 1.0))

    if iso_flag and attack_prob > 0.5:
        method = "both"
    elif iso_flag:
        method = "isolation_forest"
    else:
        method = "transformer"

    return AnomalyResult(
        timestamp        = alert.timestamp,
        source_ip        = alert.source_ip,
        dest_ip          = alert.dest_ip,
        attack_type      = alert.alert_type,
        confidence       = attack_prob,
        alert_id         = alert.alert_id,
        anomaly_score    = anomaly_score,
        is_anomalous     = is_anomalous,
        embedding        = embedding,
        detection_method = method
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("layer1_detection.server:app", host="0.0.0.0", port=8001, reload=False)
