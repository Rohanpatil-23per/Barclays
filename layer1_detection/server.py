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
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CICIDS_PATH    = os.path.join(BASE_DIR, "models/roberta_layer1")
UNSW_PATH      = os.path.join(BASE_DIR, "models/roberta_unsw")
ISO_PATH       = os.path.join(BASE_DIR, "models/isolation_forest.pkl")
CICIDS_SCALER  = os.path.join(BASE_DIR, "models/layer1_scaler.pkl")
UNSW_SCALER    = os.path.join(BASE_DIR, "models/unsw_scaler.pkl")
STATS_PATH     = os.path.join(BASE_DIR, "models/iso_forest_stats.json")
FEATS_PATH     = os.path.join(BASE_DIR, "master_dataset/top_features.json")
UNSW_FEATS     = os.path.join(BASE_DIR, "models/unsw_features.json")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN        = 128

# ── Global model state ────────────────────────────────────────────────────────
state = {}

# ── LLM Log Parser Cache ──────────────────────────────────────────────────────
from functools import lru_cache
from hashlib import sha256
import httpx
import re
import asyncio

# In-memory cache for parsed logs (max 10,000 entries)
_llm_parse_cache: dict[str, dict] = {}
_LLM_CACHE_MAX = 10000

def _cache_key(raw_log: str) -> str:
    """Generate cache key from raw log content."""
    return sha256(raw_log.encode()).hexdigest()[:16]

def fast_precheck(raw_log: str) -> dict | None:
    """
    Fast pre-check layer for trivial cases. Returns parsed dict if handled,
    None if LLM parsing is needed.
    
    Handles:
    - Empty or whitespace-only logs
    - Already valid JSON logs (including nested)
    - Simple key=value formatted logs
    """
    raw_log = raw_log.strip()
    
    # Empty log
    if not raw_log:
        return {
            "source_ip": "0.0.0.0",
            "dest_ip": "0.0.0.0",
            "protocol": "",
            "port": 0,
            "event_type": "empty",
            "alert_type": "benign",
            "severity": 0.0,
            "username": "",
            "process": "",
            "file": "",
            "privilege_level": "",
            "raw_log": raw_log,
            "_parser": "fast_precheck_empty"
        }
    
    # Already valid JSON - handle nested structures
    if raw_log.startswith("{") and raw_log.endswith("}"):
        try:
            parsed = json.loads(raw_log)
            
            # Helper to recursively find values in nested dicts
            def find_value(d, *keys):
                for key in keys:
                    if isinstance(d, dict):
                        # Direct key match
                        if key in d:
                            val = d[key]
                            if isinstance(val, dict):
                                continue  # Don't return nested dicts
                            return val
                        # Case-insensitive search
                        for k, v in d.items():
                            if k.lower() == key.lower() and not isinstance(v, dict):
                                return v
                        # Search nested dicts
                        for v in d.values():
                            if isinstance(v, dict):
                                result = find_value(v, key)
                                if result:
                                    return result
                return None
            
            # Extract fields with nested search
            source_ip = find_value(parsed, "source_ip", "src_ip", "srcip", "src", "sourceIPAddress", "client_ip", "remote_addr")
            dest_ip = find_value(parsed, "dest_ip", "dst_ip", "dstip", "dst", "destinationIPAddress", "server_ip")
            username = find_value(parsed, "username", "user", "userName", "user_name", "suser", "account")
            event_type = find_value(parsed, "event_type", "event", "type", "eventName", "action")
            
            result = {
                "source_ip": str(source_ip) if source_ip else "0.0.0.0",
                "dest_ip": str(dest_ip) if dest_ip else "0.0.0.0",
                "protocol": str(find_value(parsed, "protocol", "proto") or ""),
                "port": int(find_value(parsed, "port", "dst_port", "dport", "dpt") or 0),
                "event_type": str(event_type) if event_type else "unknown",
                "alert_type": str(find_value(parsed, "alert_type", "attack_type", "category", "threat_type") or "unknown"),
                "severity": float(find_value(parsed, "severity", "risk", "priority") or 0.5),
                "username": str(username) if username else "",
                "process": str(find_value(parsed, "process", "process_name", "cmd", "command") or ""),
                "file": str(find_value(parsed, "file", "filename", "path", "filePath") or ""),
                "privilege_level": str(find_value(parsed, "privilege_level", "priv", "privilege", "access_level") or ""),
                "raw_log": raw_log,
                "_parser": "fast_precheck_json",
                "_original": parsed
            }
            return result
        except json.JSONDecodeError:
            pass
    
    # Simple key=value format - with extended key matching
    # Matches: key=value, key:value (for CEF extensions)
    kv_pattern = re.compile(r'(\w+)=([^\s|]+)')
    matches = kv_pattern.findall(raw_log)
    
    if len(matches) >= 3:  # At least 3 key-value pairs
        kv_dict = {k.lower(): v for k, v in matches}
        
        # Extended port field matching (dpt, spt, DPT, etc.)
        port_val = (kv_dict.get("dpt") or kv_dict.get("port") or kv_dict.get("dport") or 
                   kv_dict.get("dst_port") or kv_dict.get("spt") or "0")
        try:
            port = int(port_val)
        except:
            port = 0
        
        # Extended severity detection from CEF format
        severity = 0.5
        raw_lower = raw_log.lower()
        if "|high|" in raw_lower or "|critical|" in raw_lower:
            severity = 0.9
        elif "|medium|" in raw_lower:
            severity = 0.6
        elif "|low|" in raw_lower:
            severity = 0.3
        
        # Detect attack type from CEF/log content
        alert_type = "unknown"
        if "zeus" in raw_lower or "botnet" in raw_lower:
            alert_type = "Zeus_Botnet"
        elif "scan" in raw_lower:
            alert_type = "Port_Scan"
        elif "ddos" in raw_lower:
            alert_type = "DDoS"
        elif "brute" in raw_lower or "failed password" in raw_lower:
            alert_type = "Brute_Force"
        elif "exfil" in raw_lower:
            alert_type = "Data_Exfiltration"
        elif "threat" in raw_lower:
            alert_type = "Threat_Detected"
        
        result = {
            "source_ip": kv_dict.get("src") or kv_dict.get("source_ip") or kv_dict.get("srcip") or "0.0.0.0",
            "dest_ip": kv_dict.get("dst") or kv_dict.get("dest_ip") or kv_dict.get("dstip") or "0.0.0.0",
            "protocol": kv_dict.get("proto") or kv_dict.get("protocol") or "",
            "port": port,
            "event_type": kv_dict.get("event") or kv_dict.get("event_type") or kv_dict.get("type") or kv_dict.get("action") or "unknown",
            "alert_type": alert_type,
            "severity": severity,
            "username": kv_dict.get("user") or kv_dict.get("username") or kv_dict.get("suser") or "",
            "process": kv_dict.get("process") or kv_dict.get("cmd") or "",
            "file": kv_dict.get("file") or kv_dict.get("path") or kv_dict.get("target") or "",
            "privilege_level": kv_dict.get("priv") or kv_dict.get("privilege") or "",
            "raw_log": raw_log,
            "_parser": "fast_precheck_kv"
        }
        return result
    
    # Not a trivial case - needs LLM parsing
    return None


async def llm_parse_log(raw_log: str, timeout: float = 30.0) -> dict:
    """
    Use Ollama LLM to dynamically extract structured fields from any log format.
    
    Supports: syslog, CEF, LEEF, JSON, Windows Event XML, netflow, Apache/Nginx,
    firewall logs, IDS alerts, plain text, and any other format.
    """
    prompt = f"""You are a cybersecurity log parser. Extract structured fields from the following raw log entry.

RAW LOG:
{raw_log}

Extract these fields (use empty string or 0 if not present):
- source_ip: Source IP address
- dest_ip: Destination IP address  
- protocol: Network protocol (TCP, UDP, ICMP, etc.)
- port: Destination port number
- event_type: Type of event (login, connection, alert, etc.)
- alert_type: Attack category if malicious (DDoS, SQL_Injection, Brute_Force, etc.) or "benign"
- severity: Risk level 0.0-1.0 (0=benign, 1=critical)
- username: Username involved if any
- process: Process name if any
- file: File path if any
- privilege_level: Privilege level (admin, user, root, system, etc.)

Respond ONLY with a valid JSON object, no explanation:
{{"source_ip":"...","dest_ip":"...","protocol":"...","port":0,"event_type":"...","alert_type":"...","severity":0.0,"username":"...","process":"...","file":"...","privilege_level":"..."}}"""

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent parsing
                        "num_predict": 300,  # Limit output tokens
                    }
                }
            )
            response.raise_for_status()
            
            llm_response = response.json().get("response", "")
            
            # Extract JSON from response (handle potential markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Normalize and validate
                result = {
                    "source_ip": str(parsed.get("source_ip", "0.0.0.0")) or "0.0.0.0",
                    "dest_ip": str(parsed.get("dest_ip", "0.0.0.0")) or "0.0.0.0",
                    "protocol": str(parsed.get("protocol", "")) or "",
                    "port": int(parsed.get("port", 0) or 0),
                    "event_type": str(parsed.get("event_type", "unknown")) or "unknown",
                    "alert_type": str(parsed.get("alert_type", "unknown")) or "unknown",
                    "severity": float(parsed.get("severity", 0.5) or 0.5),
                    "username": str(parsed.get("username", "")) or "",
                    "process": str(parsed.get("process", "")) or "",
                    "file": str(parsed.get("file", "")) or "",
                    "privilege_level": str(parsed.get("privilege_level", "")) or "",
                    "raw_log": raw_log,
                    "_parser": "llm"
                }
                return result
            else:
                raise ValueError(f"No JSON found in LLM response: {llm_response[:200]}")
                
    except httpx.TimeoutException:
        raise RuntimeError(f"LLM parsing timeout after {timeout}s")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LLM HTTP error: {e.response.status_code}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from LLM: {e}")


async def parse_log_dynamic(raw_log: str, use_cache: bool = True) -> dict:
    """
    Universal log parser entry point. Orchestrates fast precheck + LLM parsing.
    
    Args:
        raw_log: Raw log string in any format
        use_cache: Whether to use/update the parse cache
        
    Returns:
        Structured dict with normalized fields
    """
    global _llm_parse_cache
    
    cache_key = _cache_key(raw_log) if use_cache else None
    
    # Check cache first
    if use_cache and cache_key in _llm_parse_cache:
        cached = _llm_parse_cache[cache_key].copy()
        cached["_cache_hit"] = True
        return cached
    
    # Try fast precheck first (JSON, key=value, empty)
    fast_result = fast_precheck(raw_log)
    if fast_result is not None:
        if use_cache and cache_key:
            # Evict oldest if cache full
            if len(_llm_parse_cache) >= _LLM_CACHE_MAX:
                oldest_key = next(iter(_llm_parse_cache))
                del _llm_parse_cache[oldest_key]
            _llm_parse_cache[cache_key] = fast_result
        return fast_result
    
    # Fall back to LLM parsing
    llm_result = await llm_parse_log(raw_log)
    
    if use_cache and cache_key:
        if len(_llm_parse_cache) >= _LLM_CACHE_MAX:
            oldest_key = next(iter(_llm_parse_cache))
            del _llm_parse_cache[oldest_key]
        _llm_parse_cache[cache_key] = llm_result
    
    return llm_result

# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading models on {DEVICE}...")

    # CICIDS RoBERTa
    state["cicids_tok"] = RobertaTokenizer.from_pretrained(CICIDS_PATH, local_files_only=True)
    state["cicids_rob"] = RobertaForSequenceClassification.from_pretrained(CICIDS_PATH, local_files_only=True)
    state["cicids_rob"].to(DEVICE)
    state["cicids_rob"].eval()

    # UNSW RoBERTa
    state["unsw_tok"] = RobertaTokenizer.from_pretrained(UNSW_PATH, local_files_only=True)
    state["unsw_rob"] = RobertaForSequenceClassification.from_pretrained(UNSW_PATH, local_files_only=True)
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

    # ── Handle missing features - generate from alert metadata ────────────────
    if alert.features is None or len(alert.features) < 77:
        # Generate 77 synthetic features from available alert data
        # This allows /detect to work with simple alert JSON without full CICIDS features
        base_features = [0.0] * 77
        
        # Protocol encoding (index 0)
        proto_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        base_features[0] = float(proto_map.get(str(alert.protocol or "").upper(), 0))
        
        # Port as proxy for some features
        port_val = float(alert.port or 0)
        base_features[4] = port_val  # Can represent flow characteristic
        
        # Severity-based feature injection
        sev_map = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.2}
        sev_score = sev_map.get(str(alert.severity or "medium").lower(), 0.5)
        
        # Attack type influences anomaly indicators
        attack_lower = str(alert.alert_type or "").lower()
        if "ddos" in attack_lower:
            base_features[2] = 10000.0   # Total Fwd Packets (high)
            base_features[14] = 1500000.0  # Flow Bytes/s (high)
            base_features[44] = 800.0    # SYN Flag Count (high)
        elif "scan" in attack_lower or "portscan" in attack_lower:
            base_features[44] = 500.0    # SYN Flag Count
            base_features[15] = 800.0    # Flow Packets/s
        elif "brute" in attack_lower:
            base_features[2] = 100.0     # Total Fwd Packets
            base_features[3] = 100.0     # Total Backward Packets
        elif "zeus" in attack_lower or "botnet" in attack_lower:
            base_features[14] = 50000.0  # Flow Bytes/s
            base_features[2] = 500.0     # Total Fwd Packets
        elif "sql" in attack_lower or "injection" in attack_lower:
            base_features[4] = 3306.0    # MySQL port
            base_features[14] = 10000.0  # Flow Bytes/s
        
        # Blend in any partial features provided
        if alert.features:
            for i, f in enumerate(alert.features[:77]):
                if f is not None:
                    base_features[i] = float(f)
        
        features = np.array(base_features).reshape(1, -1)
    else:
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

    # ── Pass raw CICIDS scaled features to L4 (77 features for 77-feature model) ────
    # FIX 5/6: Layer 4 now uses 77 features, not 25
    cicids_features_77 = scaled[0][:77].tolist() if len(scaled[0]) >= 77 else (
        scaled[0].tolist() + [0.0] * (77 - len(scaled[0]))
    )

    result = AnomalyResult(
        alert_id         = alert.alert_id,
        timestamp        = alert.timestamp,
        source_ip        = alert.source_ip,
        dest_ip          = alert.dest_ip,
        attack_type      = effective_attack_type,
        anomaly_score    = anomaly_score,
        is_anomalous     = is_anomalous,
        embedding        = embedding,
        roberta_embedding = embedding,   # 768D RoBERTa CLS — explicit field for L2→L3 carry-through
        detection_method = method,
        confidence       = attack_prob,
        cicids_features  = cicids_features_77,  # FIX 6: 77 features for L4
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

# ── Raw Log Detection Endpoint (LLM-powered) ──────────────────────────────────
from pydantic import BaseModel as PydanticBaseModel

class RawLogRequest(PydanticBaseModel):
    raw_log: str
    use_cache: bool = True

class RawLogResult(PydanticBaseModel):
    alert_id: str
    raw_log: str
    parsed_fields: dict
    parser_used: str
    anomaly_score: float
    is_anomalous: bool
    detection_method: str
    confidence: float
    attack_type: str
    source_ip: str
    dest_ip: str

@app.post("/detect_raw", response_model=RawLogResult)
async def detect_raw(request: RawLogRequest):
    """
    Accept a raw log in ANY format and detect anomalies using LLM parsing.
    
    Supports: syslog, CEF, LEEF, JSON, Windows Event XML, netflow, 
    Apache/Nginx access logs, firewall logs, IDS alerts, plain text, etc.
    
    The log is first run through fast_precheck (handles JSON, key=value, empty).
    If not trivial, Ollama llama3.1:8b parses it dynamically.
    """
    import uuid
    from datetime import datetime
    
    raw_log = request.raw_log
    
    # Parse the raw log using LLM-powered parser
    try:
        parsed = await parse_log_dynamic(raw_log, use_cache=request.use_cache)
    except Exception as e:
        raise HTTPException(500, f"Log parsing failed: {e}")
    
    parser_used = parsed.get("_parser", "unknown")
    cache_hit = parsed.get("_cache_hit", False)
    
    # Generate alert ID
    alert_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    # Extract fields for detection
    source_ip = parsed.get("source_ip", "0.0.0.0")
    dest_ip = parsed.get("dest_ip", "0.0.0.0")
    alert_type = parsed.get("alert_type", "unknown")
    severity = parsed.get("severity", 0.5)
    
    # Check IOC list
    ioc_type = state["redis"].is_ioc(source_ip)
    
    # Since we don't have numeric features from raw logs, use RoBERTa on raw text
    # This is the key insight: RoBERTa can score raw log text directly
    cicids_prob, embedding = _roberta_score(
        state["cicids_rob"], state["cicids_tok"], raw_log[:512]  # Truncate to model max
    )
    
    # Also score with UNSW RoBERTa for ensemble
    unsw_prob, _ = _roberta_score(
        state["unsw_rob"], state["unsw_tok"], raw_log[:512]
    )
    attack_prob = 0.6 * cicids_prob + 0.4 * unsw_prob
    
    # FAISS similarity check
    faiss_anomalous, faiss_score = state["faiss"].is_anomalous(embedding)
    
    # IOC severity boost
    if ioc_type:
        attack_prob = min(1.0, attack_prob + 0.2)
    
    # LLM-parsed severity boost (if LLM says it's malicious)
    if severity > 0.7 and parser_used == "llm":
        attack_prob = min(1.0, attack_prob + 0.15)
    
    # Decision fusion
    is_anomalous = attack_prob > 0.5 or faiss_anomalous
    anomaly_score = float(np.clip(
        max(attack_prob, float(faiss_anomalous) * 0.7, severity * 0.6),
        0.0, 1.0
    ))
    
    if faiss_anomalous and attack_prob > 0.5:
        method = "llm+faiss+ensemble"
    elif faiss_anomalous:
        method = "llm+faiss"
    else:
        method = "llm+ensemble"
    
    if cache_hit:
        method += "+cached"
    
    # Effective attack type
    effective_attack_type = alert_type
    if alert_type in ("unknown", "benign") and attack_prob > 0.5:
        effective_attack_type = "Suspicious_Activity"
    
    result = RawLogResult(
        alert_id=alert_id,
        raw_log=raw_log[:500],  # Truncate for response
        parsed_fields=parsed,
        parser_used=parser_used,
        anomaly_score=round(anomaly_score, 4),
        is_anomalous=is_anomalous,
        detection_method=method,
        confidence=round(attack_prob, 4),
        attack_type=effective_attack_type,
        source_ip=source_ip,
        dest_ip=dest_ip,
    )
    
    # Cache in Redis
    state["redis"].cache_anomaly(alert_id, {
        "alert_id": alert_id,
        "source_ip": source_ip,
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous,
        "method": method,
    })
    state["redis"].cache_embedding(alert_id, embedding)
    
    # Publish if anomalous
    if is_anomalous:
        state["kafka"].send("anomaly_results", {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "attack_type": effective_attack_type,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "detection_method": method,
            "raw_log": raw_log[:500],
        }, key=alert_id)
    
    return result


@app.post("/detect_raw/batch")
async def detect_raw_batch(logs: list[str], use_cache: bool = True):
    """
    Batch process multiple raw logs concurrently with LLM parsing.
    
    Args:
        logs: List of raw log strings (max 1000)
        use_cache: Whether to use parse cache
        
    Returns:
        List of detection results with throughput metrics
    """
    import time
    
    if not logs:
        return {"total": 0, "results": [], "processing_time_ms": 0, "throughput": 0}
    
    logs = logs[:1000]  # Limit
    n = len(logs)
    start = time.perf_counter()
    
    # Process all logs concurrently
    semaphore = asyncio.Semaphore(32)  # Limit concurrent LLM calls
    
    async def process_one(raw_log: str):
        async with semaphore:
            try:
                req = RawLogRequest(raw_log=raw_log, use_cache=use_cache)
                return await detect_raw(req)
            except Exception as e:
                return {"error": str(e), "raw_log": raw_log[:100]}
    
    tasks = [process_one(log) for log in logs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert results
    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append({"error": str(r)})
        elif isinstance(r, RawLogResult):
            processed.append(r.model_dump())
        elif isinstance(r, dict):
            processed.append(r)
        else:
            processed.append({"error": "unexpected result type"})
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    throughput = round(n / max(elapsed_ms / 1000, 0.001), 1)
    
    anomalous_count = sum(1 for r in processed if r.get("is_anomalous"))
    
    return {
        "total": n,
        "anomalous": anomalous_count,
        "processing_time_ms": round(elapsed_ms, 2),
        "throughput": throughput,
        "results": processed,
    }

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

# ── Batch Ingestion Endpoint ──────────────────────────────────────────────────
import asyncio
import time
import logging
logger = logging.getLogger("layer1")

from pydantic import BaseModel as PydanticBaseModel

class BatchIngestAlert(PydanticBaseModel):
    alert_id:    str = ""
    timestamp:   str = ""
    source_ip:   str = "0.0.0.0"
    dest_ip:     str = "0.0.0.0"
    alert_type:  str = "unknown"
    severity:    float = 0.5
    features:    list = []
    event_type:  str = ""
    protocol:    str = ""
    port:        int = 0
    username:    str = ""
    process:     str = ""
    file:        str = ""
    privilege_level: str = ""
    source_type: str = "siem"
    raw_event:   dict = {}

class BatchIngestRequest(PydanticBaseModel):
    alerts: list[BatchIngestAlert]

class BatchIngestResponse(PydanticBaseModel):
    total: int
    anomalous: int
    rejected_by_relevance: int
    processing_time_ms: float
    throughput: float
    results: list[dict]


async def _process_single_ingest(
    alert_data: BatchIngestAlert, 
    state: dict, 
    relevance_analyzer,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process a single alert through the full pipeline."""
    async with semaphore:
        try:
            from layer1_detection.ingestion import EventNormalizer
            from shared.schemas import Alert
            import uuid
            from datetime import datetime
            
            # Normalize if raw_event provided, otherwise use direct fields
            if alert_data.raw_event:
                norm = EventNormalizer()
                normalized = norm.normalize(alert_data.raw_event, alert_data.source_type)
                if normalized:
                    alert = normalized
                else:
                    # Fallback to direct fields
                    alert = Alert(
                        alert_id=alert_data.alert_id or str(uuid.uuid4()),
                        timestamp=alert_data.timestamp or datetime.utcnow().isoformat(),
                        source_ip=alert_data.source_ip,
                        dest_ip=alert_data.dest_ip,
                        alert_type=alert_data.alert_type,
                        severity=alert_data.severity,
                        features=alert_data.features or [0.0] * 77,
                        event_type=alert_data.event_type,
                        protocol=alert_data.protocol,
                        port=alert_data.port,
                        username=alert_data.username,
                        process=alert_data.process,
                        file=alert_data.file,
                        privilege_level=alert_data.privilege_level,
                    )
            else:
                alert = Alert(
                    alert_id=alert_data.alert_id or str(uuid.uuid4()),
                    timestamp=alert_data.timestamp or datetime.utcnow().isoformat(),
                    source_ip=alert_data.source_ip,
                    dest_ip=alert_data.dest_ip,
                    alert_type=alert_data.alert_type,
                    severity=alert_data.severity,
                    features=alert_data.features or [0.0] * 77,
                    event_type=alert_data.event_type,
                    protocol=alert_data.protocol,
                    port=alert_data.port,
                    username=alert_data.username,
                    process=alert_data.process,
                    file=alert_data.file,
                    privilege_level=alert_data.privilege_level,
                )
            
            # Ensure 77 features
            features = list(alert.features)
            if len(features) < 77:
                features.extend([0.0] * (77 - len(features)))
            features = features[:77]
            
            features_np = np.array(features).reshape(1, -1)
            
            # Isolation Forest
            scaled = state["cicids_scaler"].transform(features_np)
            iso_score = float(state["iso"].decision_function(scaled)[0])
            iso_flag = iso_score < state["iso_threshold"]
            
            # RoBERTa (individual call - batch version handles this more efficiently)
            from layer1_detection.inference_utils import serialize_features
            cicids_text = serialize_features(features, state["cicids_feats"])
            attack_prob, embedding = _roberta_score(
                state["cicids_rob"], state["cicids_tok"], cicids_text
            )
            
            # FAISS
            emb_np = np.array([embedding], dtype=np.float32)
            D, _ = state["faiss"].index.search(emb_np, 5)
            faiss_anomalous = float(D[0].min()) > getattr(state["faiss"], 'threshold', 0.7)
            
            # Decision fusion
            is_anomalous = attack_prob > 0.5 or iso_flag or faiss_anomalous
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
            
            # Relevance check
            relevance_result = relevance_analyzer.analyze(
                features=features,
                prediction={
                    "anomaly_score": anomaly_score,
                    "is_anomalous": is_anomalous,
                    "confidence": attack_prob,
                },
                source_ip=alert.source_ip
            )
            
            result = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "source_ip": alert.source_ip,
                "dest_ip": alert.dest_ip,
                "attack_type": alert.alert_type,
                "anomaly_score": round(anomaly_score, 4),
                "is_anomalous": is_anomalous,
                "detection_method": method,
                "confidence": round(attack_prob, 4),
                "relevance_passes": relevance_result["passes"],
                "relevance_score": relevance_result["relevance_score"],
                "rejection_reason": relevance_result["rejection_reason"],
            }
            
            # Publish to Kafka if anomalous and passes relevance
            if is_anomalous and relevance_result["passes"]:
                state["kafka"].send("immunex_alerts", result, key=alert.alert_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
            return {
                "alert_id": alert_data.alert_id or "unknown",
                "error": str(e),
                "is_anomalous": False,
                "relevance_passes": False,
            }


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(request: BatchIngestRequest):
    """
    Batch ingest endpoint for high-throughput log processing.
    
    Accepts up to 10,000 alerts and processes them concurrently.
    Each alert goes through: normalization → RoBERTa → Isolation Forest → 
    FAISS → RelevanceAnalyzer → Kafka publish if anomalous.
    
    Returns summary with throughput metrics.
    """
    if not request.alerts:
        return BatchIngestResponse(
            total=0, anomalous=0, rejected_by_relevance=0,
            processing_time_ms=0, throughput=0, results=[]
        )
    
    # Limit to 10,000 alerts
    alerts = request.alerts[:10000]
    n = len(alerts)
    
    start_time = time.perf_counter()
    
    # Initialize relevance analyzer
    from layer1_detection.relevance_analyzer import RelevanceAnalyzer
    relevance_analyzer = RelevanceAnalyzer(device=str(DEVICE))
    
    # Semaphore for concurrent processing (max 64)
    semaphore = asyncio.Semaphore(64)
    
    # Process all alerts concurrently
    tasks = [
        _process_single_ingest(alert, state, relevance_analyzer, semaphore)
        for alert in alerts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for r in results:
        if isinstance(r, Exception):
            processed_results.append({
                "error": str(r),
                "is_anomalous": False,
                "relevance_passes": False,
            })
        else:
            processed_results.append(r)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    throughput = round(n / max(elapsed_ms / 1000, 0.001), 1)
    
    anomalous_count = sum(1 for r in processed_results if r.get("is_anomalous"))
    rejected_count = sum(1 for r in processed_results 
                         if r.get("is_anomalous") and not r.get("relevance_passes"))
    
    logger.info(
        f"[BATCH INGEST] n={n} | {elapsed_ms:.0f}ms | {throughput} logs/s | "
        f"anomalous={anomalous_count} rejected={rejected_count}"
    )
    
    return BatchIngestResponse(
        total=n,
        anomalous=anomalous_count,
        rejected_by_relevance=rejected_count,
        processing_time_ms=round(elapsed_ms, 2),
        throughput=throughput,
        results=processed_results,
    )
from layer1_detection.batch_endpoint import create_batch_detect_endpoint
create_batch_detect_endpoint(app, state)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("layer1_detection.server:app", host="0.0.0.0", port=8001, reload=False)
