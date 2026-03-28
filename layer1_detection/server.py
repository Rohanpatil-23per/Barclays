import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import torch
import requests as req
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from shared.schemas import Alert, AnomalyResult
from shared.kafka_client import IMMUNEXProducer
from shared.redis_client import IMMUNEXCache
from shared.es_client import IMMUNEXElastic
from layer1_detection.inference_utils import load_top_features, serialize_features
from layer1_detection.faiss_index import FAISSIndex

# ── Config ────────────────────────────────────────────────────────────────────
CICIDS_PATH    = "models/roberta_layer1"
UNSW_PATH      = "models/roberta_unsw"
ISO_PATH       = "models/isolation_forest.pkl"
CICIDS_SCALER  = "models/layer1_scaler.pkl"
UNSW_SCALER    = "models/unsw_scaler.pkl"
STATS_PATH     = "models/iso_forest_stats.json"
FEATS_PATH     = "master_dataset/top_features.json"
UNSW_FEATS     = "models/unsw_features.json"
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN        = 128

# ── Global model state ────────────────────────────────────────────────────────
state = {}

# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading models on {DEVICE}...")

    # CICIDS RoBERTa
    state["cicids_tok"] = RobertaTokenizer.from_pretrained(CICIDS_PATH)
    state["cicids_rob"] = RobertaForSequenceClassification.from_pretrained(CICIDS_PATH)
    state["cicids_rob"].to(DEVICE)
    state["cicids_rob"].eval()

    # UNSW RoBERTa
    state["unsw_tok"] = RobertaTokenizer.from_pretrained(UNSW_PATH)
    state["unsw_rob"] = RobertaForSequenceClassification.from_pretrained(UNSW_PATH)
    state["unsw_rob"].to(DEVICE)
    state["unsw_rob"].eval()

    # Isolation Forest
    with open(ISO_PATH, "rb") as f:
        state["iso"] = pickle.load(f)
    with open(CICIDS_SCALER, "rb") as f:
        state["cicids_scaler"] = pickle.load(f)
    with open(UNSW_SCALER, "rb") as f:
        state["unsw_scaler"] = pickle.load(f)

    # Features
    state["cicids_feats"] = load_top_features(FEATS_PATH)
    with open(UNSW_FEATS) as f:
        state["unsw_feats"] = json.load(f)

    # Threshold
    with open(STATS_PATH) as f:
        state["iso_threshold"] = json.load(f)["threshold"]

    # Infrastructure
    state["kafka"]  = IMMUNEXProducer()
    state["redis"]  = IMMUNEXCache()
    state["es"]     = IMMUNEXElastic()
    state["faiss"]  = FAISSIndex(use_gpu=torch.cuda.is_available())

    print("All models loaded ✅")
    print(f"  CICIDS RoBERTa: {CICIDS_PATH}")
    print(f"  UNSW RoBERTa:   {UNSW_PATH}")
    print(f"  Isolation Forest: loaded")
    print(f"  FAISS GPU index: {state['faiss'].index.ntotal} vectors")
    print(f"  Kafka/Redis/ES:  connected")

    yield

    # Shutdown
    state["kafka"].close()
    print("Server shutdown complete")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IMMUNEX Layer 1 — Innate Detection",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _roberta_score(model, tokenizer, text):
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        out         = model(
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE)
        )
        attack_prob = float(torch.softmax(out.logits, dim=1)[0][1].cpu())
        embedding   = model.roberta(
            input_ids      = enc["input_ids"].to(DEVICE),
            attention_mask = enc["attention_mask"].to(DEVICE)
        ).last_hidden_state[:, 0, :].squeeze().cpu().tolist()
    return attack_prob, embedding

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "layer":   1,
        "device":  str(DEVICE),
        "models":  ["roberta_cicids", "roberta_unsw", "isolation_forest", "llama3.1:8b"],
        "faiss_vectors": state.get("faiss", {}).index.ntotal if "faiss" in state else 0,
    }

@app.post("/detect", response_model=AnomalyResult)
async def detect(alert: Alert):
    # ── Check Redis cache first ───────────────────────────────────────────────
    cached = state["redis"].get_anomaly(alert.alert_id)
    if cached:
        return AnomalyResult(**cached)

    # ── Check IOC list ────────────────────────────────────────────────────────
    ioc_type = state["redis"].is_ioc(alert.source_ip)

    features = np.array(alert.features).reshape(1, -1)

    # ── Isolation Forest ──────────────────────────────────────────────────────
    scaled    = state["cicids_scaler"].transform(features)
    iso_score = float(state["iso"].decision_function(scaled)[0])
    iso_flag  = iso_score < state["iso_threshold"]

    # ── CICIDS RoBERTa ────────────────────────────────────────────────────────
    cicids_text          = serialize_features(alert.features, state["cicids_feats"])
    cicids_prob, embedding = _roberta_score(
        state["cicids_rob"], state["cicids_tok"], cicids_text
    )

    # ── UNSW RoBERTa ensemble ─────────────────────────────────────────────────
    unsw_arr  = state["unsw_scaler"].transform(features[:, :39])
    unsw_text = " ".join(
        f"{n.lower().replace(' ','_')}:{unsw_arr[0][i]:.4f}"
        for i, n in enumerate(state["unsw_feats"][:25])
    )
    unsw_prob, _ = _roberta_score(
        state["unsw_rob"], state["unsw_tok"], unsw_text
    )
    attack_prob = 0.6 * cicids_prob + 0.4 * unsw_prob

    # ── FAISS similarity check ────────────────────────────────────────────────
    faiss_anomalous, faiss_score = state["faiss"].is_anomalous(embedding)

    # ── IOC severity boost ────────────────────────────────────────────────────
    if ioc_type:
        attack_prob = min(1.0, attack_prob + 0.2)

    # ── Decision fusion ───────────────────────────────────────────────────────
    is_anomalous  = attack_prob > 0.5 or iso_flag or faiss_anomalous
    anomaly_score = float(np.clip(
        max(attack_prob, float(iso_flag) * 0.8, float(faiss_anomalous) * 0.7),
        0.0, 1.0
    ))

    if iso_flag and attack_prob > 0.5:
        method = "both"
    elif iso_flag:
        method = "isolation_forest"
    elif faiss_anomalous:
        method = "faiss_similarity"
    else:
        method = "ensemble"

    # ── Novel attack detection: low RoBERTa conf but high iso/faiss score ──────
    effective_attack_type = alert.alert_type
    if attack_prob < 0.15 and anomaly_score > 0.6:
        effective_attack_type = "Unknown_Novel_Attack"
        method = method + "+novel_flag"

    # ── Pass raw CICIDS scaled features to L4 (semantic signal) ─────────────
    cicids_features_25 = scaled[0][:25].tolist()

    result = AnomalyResult(
        alert_id         = alert.alert_id,
        timestamp        = alert.timestamp,
        source_ip        = alert.source_ip,
        dest_ip          = alert.dest_ip,
        attack_type      = effective_attack_type,
        anomaly_score    = anomaly_score,
        is_anomalous     = is_anomalous,
        embedding        = embedding,
        detection_method = method,
        confidence       = attack_prob,
        cicids_features  = cicids_features_25,
        event_type       = alert.event_type,
        protocol         = alert.protocol,
        port             = alert.port,
        username         = alert.username,
        process          = alert.process,
        file             = alert.file,
        privilege_level  = alert.privilege_level,
    )

    # ── Publish to Kafka → Layer 2 ────────────────────────────────────────────
    if is_anomalous:
        state["kafka"].send("anomaly_results", result.dict(), key=alert.alert_id)

    # ── Cache in Redis ────────────────────────────────────────────────────────
    state["redis"].cache_anomaly(alert.alert_id, result.dict())
    state["redis"].cache_embedding(alert.alert_id, embedding)
    state["redis"].increment_alert_count(alert.source_ip)

    # ── Log to Elasticsearch ──────────────────────────────────────────────────
    try:
        state["es"].index_alert({
        "alert_id":      alert.alert_id,
        "source_ip":     alert.source_ip,
        "dest_ip":       alert.dest_ip,
        "attack_type":   alert.alert_type,
        "anomaly_score": anomaly_score,
        "is_anomalous":  is_anomalous,
        "layer":         1,
        "method":        method,
        "ioc_match":     ioc_type or "",
        })
    except Exception as es_err:
        logger.warning(f"ES index_alert failed (non-fatal): {es_err}")

    # ── Pub/sub for live dashboard ────────────────────────────────────────────
    state["redis"].publish("immunex_live", {
        "alert_id":      alert.alert_id,
        "anomaly_score": anomaly_score,
        "is_anomalous":  is_anomalous,
        "source_ip":     alert.source_ip,
        "method":        method,
    })

    return result

@app.post("/ingest")
async def ingest(raw_event: dict):
    """
    Accept raw events from SIEM/EDR/Network/Auth sources.
    Normalizes and publishes to Kafka immunex_raw_alerts topic.
    """
    from layer1_detection.ingestion import EventNormalizer
    source_type = raw_event.pop("source_type", "siem")
    norm  = EventNormalizer()
    alert = norm.normalize(raw_event, source_type)
    if not alert:
        raise HTTPException(400, f"Could not normalize event of type: {source_type}")
    state["kafka"].send("raw_alerts", alert.dict(), key=alert.alert_id)
    return {"status": "ingested", "alert_id": alert.alert_id}

@app.post("/generate-playbook")
async def generate_playbook(data: dict):
    """
    Wraps Ollama so all layers call this instead of Ollama directly.
    Input:  {"prompt": "...", "context": {...}}
    Output: {"response": "...", "status": "ok"}
    """
    prompt  = data.get("prompt", "")
    context = data.get("context", {})

    full_prompt = f"""You are IMMUNEX, an autonomous cyber defense system for banking infrastructure.

Security incident context:
{json.dumps(context, indent=2)}

Task: {prompt}

Respond with a concise, actionable security playbook. Be specific about IPs, attack types, and response steps."""

    try:
        r = req.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
            timeout=60
        )
        return {"response": r.json().get("response", ""), "status": "ok"}
    except Exception as e:
        return {"response": "", "status": "error", "detail": str(e)}

@app.get("/stats")
def stats():
    """Live stats for dashboard."""
    return {
        "layer":         1,
        "faiss_vectors": state["faiss"].index.ntotal,
        "device":        str(DEVICE),
        "kafka_topics":  ["immunex_raw_alerts", "immunex_anomaly_results",
                          "immunex_attack_graphs", "immunex_responses", "immunex_playbooks"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("layer1_detection.server:app", host="0.0.0.0", port=8001, reload=False)
