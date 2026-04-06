import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, uuid, asyncio, logging, hashlib, time as _time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict
from collections import defaultdict
import threading

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from shared.schemas import Alert, AnomalyResult, AttackGraph, ResponseAction, Playbook
from shared.kafka_client import IMMUNEXProducer
from shared.redis_client import IMMUNEXCache
from shared.es_client import IMMUNEXElastic
from orchestrator.ingest_api import router as ingest_router, init_ingest
from orchestrator.security_status import router as security_router


# ── FIX 9: Production-grade structured logging ─────────────────────────────────
class JSONFormatter(logging.Formatter):
    """JSON log formatter for production observability."""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'pipeline_id'):
            log_data["pipeline_id"] = record.pipeline_id
        if hasattr(record, 'layer'):
            log_data["layer"] = record.layer
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Use JSON logging in production, standard format in dev
_log_format = os.getenv("LOG_FORMAT", "standard")
if _log_format == "json":
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
logger = logging.getLogger("orchestrator")

# ── Layer URLs with fallbacks ─────────────────────────────────────────────────
# Primary → Fallback order follows GPU power ranking
LAYER_URLS = {
    1: [
        os.getenv("LAYER1_URL",    "http://localhost:8001"),
    ],
    2: [
        os.getenv("LAYER2_URL",    "http://192.168.137.1:8002"),   # Acer Nitro 4050
        os.getenv("LAYER2_FB1",    "http://192.168.137.213:8002"),   # Lenovo LOQ 3050
        os.getenv("LAYER2_FB2",    "http://192.168.137.225:8002"),   # HP Victus 2050
        os.getenv("LAYER2_FB3",    "http://localhost:8002"),   # your machine last
    ],
    3: [
        os.getenv("LAYER3_URL",    "http://192.168.137.213:8003"),   # Lenovo LOQ 3050
        os.getenv("LAYER3_FB1",    "http://192.168.137.1:8003"),   # Acer Nitro 4050
        os.getenv("LAYER3_FB2",    "http://192.168.137.225:8003"),   # HP Victus 2050
        os.getenv("LAYER3_FB3",    "http://localhost:8003"),
    ],
    4: [
        os.getenv("LAYER4_URL",    "http://192.168.137.225:8004"),   # HP Victus 2050
        os.getenv("LAYER4_FB1",    "http://192.168.137.244:8004"),   # HP Pavilion 1650
        os.getenv("LAYER4_FB2",    "http://192.168.137.1:8004"),   # Acer Nitro 4050
        os.getenv("LAYER4_FB3",    "http://localhost:8004"),
    ],
    5: [
        os.getenv("LAYER5_URL",    "http://192.168.137.244:8005"),   # HP Pavilion 1650
        os.getenv("LAYER5_FB1",    "http://192.168.137.213:8005"),   # Lenovo LOQ 3050
        os.getenv("LAYER5_FB2",    "http://192.168.137.1:8005"),   # Acer Nitro 4050
        os.getenv("LAYER5_FB3",    "http://localhost:8005"),
    ],
}

# ── FIX 9: Production configuration ───────────────────────────────────────────
TIMEOUT = float(os.getenv("ORCHESTRATOR_TIMEOUT", "30.0"))
LAYER_TIMEOUTS = {  # Per-layer timeouts (seconds)
    1: float(os.getenv("L1_TIMEOUT", "15.0")),  # Detection - fast
    2: float(os.getenv("L2_TIMEOUT", "20.0")),  # Correlation - moderate
    3: float(os.getenv("L3_TIMEOUT", "30.0")),  # Response - may call LLM
    4: float(os.getenv("L4_TIMEOUT", "15.0")),  # Immunity - fast
    5: float(os.getenv("L5_TIMEOUT", "20.0")),  # Threat memory - moderate
}

state = {}

# ── Track which URL is currently active per layer ────────────────────────────
_active_url: dict = {}


# ── FIX 9: Circuit Breaker for layer resilience ───────────────────────────────
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures: Dict[str, int] = defaultdict(int)
        self._state: Dict[str, str] = defaultdict(lambda: "CLOSED")  # CLOSED, OPEN, HALF_OPEN
        self._last_failure: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_success(self, key: str):
        with self._lock:
            self._failures[key] = 0
            self._state[key] = "CLOSED"

    def record_failure(self, key: str):
        with self._lock:
            self._failures[key] += 1
            self._last_failure[key] = _time.time()
            if self._failures[key] >= self.failure_threshold:
                self._state[key] = "OPEN"
                logger.warning(f"Circuit OPEN for {key} after {self._failures[key]} failures")

    def can_proceed(self, key: str) -> bool:
        with self._lock:
            state = self._state[key]
            if state == "CLOSED":
                return True
            if state == "OPEN":
                # Check if recovery timeout has passed
                last = self._last_failure.get(key, 0)
                if _time.time() - last > self.recovery_timeout:
                    self._state[key] = "HALF_OPEN"
                    logger.info(f"Circuit HALF_OPEN for {key} - testing recovery")
                    return True
                return False
            # HALF_OPEN - allow one test request
            return True

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "states": dict(self._state),
                "failures": dict(self._failures),
            }

circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)


# ── FIX 9: Rate Limiter ───────────────────────────────────────────────────────
class RateLimiter:
    """Token bucket rate limiter for pipeline requests."""
    def __init__(self, rate: float = 100.0, capacity: float = 200.0):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = _time.time()
        self._lock = threading.Lock()
        self._total_requests = 0
        self._rejected_requests = 0

    def acquire(self) -> bool:
        with self._lock:
            now = _time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now
            self._total_requests += 1

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            self._rejected_requests += 1
            return False

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "tokens_available": round(self._tokens, 2),
                "total_requests": self._total_requests,
                "rejected_requests": self._rejected_requests,
                "rate_per_sec": self.rate,
            }

rate_limiter = RateLimiter(
    rate=float(os.getenv("RATE_LIMIT_RPS", "100")),
    capacity=float(os.getenv("RATE_LIMIT_BURST", "200"))
)


# ── FIX 9: Metrics tracking ───────────────────────────────────────────────────
class Metrics:
    """Simple metrics collector for production observability."""
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._start_time = _time.time()

    def inc(self, name: str, value: int = 1):
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float):
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def get_summary(self) -> Dict:
        with self._lock:
            summary = {
                "uptime_seconds": round(_time.time() - self._start_time, 2),
                "counters": dict(self._counters),
                "histograms": {}
            }
            for name, values in self._histograms.items():
                if values:
                    sorted_v = sorted(values)
                    n = len(sorted_v)
                    summary["histograms"][name] = {
                        "count": n,
                        "min": round(min(sorted_v), 3),
                        "max": round(max(sorted_v), 3),
                        "mean": round(sum(sorted_v) / n, 3),
                        "p50": round(sorted_v[n // 2], 3),
                        "p95": round(sorted_v[int(n * 0.95)], 3) if n > 20 else None,
                        "p99": round(sorted_v[int(n * 0.99)], 3) if n > 100 else None,
                    }
            return summary

metrics = Metrics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["kafka"] = IMMUNEXProducer()
    state["redis"] = IMMUNEXCache()
    state["es"]    = IMMUNEXElastic()
    state["http"]  = httpx.AsyncClient(timeout=TIMEOUT)
    # Initialize active URLs to primary
    for num, urls in LAYER_URLS.items():
        _active_url[num] = urls[0]
    logger.info("Orchestrator started")
    init_ingest(state)
    logger.info(f"Layer primary URLs: { {k: v[0] for k,v in LAYER_URLS.items()} }")
    yield
    await state["http"].aclose()
    state["kafka"].close()
    logger.info("Orchestrator shutdown")

app = FastAPI(title="IMMUNEX Orchestrator", version="2.0", lifespan=lifespan)
app.include_router(security_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(ingest_router)


# ── FIX 9: Call layer with circuit breaker, metrics, and per-layer timeouts ───
async def call_layer(layer_num: int, endpoint: str, payload: dict, 
                     pipeline_id: str = None) -> Optional[dict]:
    """
    Call a layer with automatic failover, circuit breaker, and metrics.
    
    - Uses per-layer timeouts from LAYER_TIMEOUTS
    - Records metrics for latency tracking
    - Uses circuit breaker to prevent cascading failures
    """
    urls = LAYER_URLS[layer_num]
    layer_timeout = LAYER_TIMEOUTS.get(layer_num, TIMEOUT)
    
    for url in urls:
        circuit_key = f"layer{layer_num}:{url}"
        
        # Check circuit breaker
        if not circuit_breaker.can_proceed(circuit_key):
            logger.warning(f"Circuit OPEN for {circuit_key} — skipping")
            metrics.inc(f"layer{layer_num}_circuit_open")
            continue
        
        full_url = f"{url}{endpoint}"
        start_time = _time.time()
        
        try:
            r = await state["http"].post(full_url, json=payload, timeout=layer_timeout)
            r.raise_for_status()
            
            # Record success
            duration_ms = (_time.time() - start_time) * 1000
            circuit_breaker.record_success(circuit_key)
            metrics.inc(f"layer{layer_num}_success")
            metrics.observe(f"layer{layer_num}_latency_ms", duration_ms)
            
            if url != _active_url[layer_num]:
                logger.warning(f"Layer {layer_num} FAILOVER active: {url}")
                _active_url[layer_num] = url
            else:
                logger.info(f"L{layer_num} {endpoint} OK ({duration_ms:.0f}ms) [pid={pipeline_id}]")
            
            return r.json()
            
        except httpx.ConnectError:
            circuit_breaker.record_failure(circuit_key)
            metrics.inc(f"layer{layer_num}_connect_error")
            logger.warning(f"L{layer_num} OFFLINE at {url} — trying next [pid={pipeline_id}]")
            continue
        except httpx.TimeoutException:
            circuit_breaker.record_failure(circuit_key)
            metrics.inc(f"layer{layer_num}_timeout")
            logger.warning(f"L{layer_num} TIMEOUT ({layer_timeout}s) at {url} [pid={pipeline_id}]")
            continue
        except httpx.HTTPStatusError as e:
            circuit_breaker.record_failure(circuit_key)
            metrics.inc(f"layer{layer_num}_http_error")
            logger.error(f"L{layer_num} HTTP {e.response.status_code} at {url} [pid={pipeline_id}]")
            continue
        except Exception as e:
            circuit_breaker.record_failure(circuit_key)
            metrics.inc(f"layer{layer_num}_error")
            logger.error(f"L{layer_num} ERROR at {url}: {e} [pid={pipeline_id}]")
            continue
    
    metrics.inc(f"layer{layer_num}_all_failed")
    logger.error(f"L{layer_num} ALL URLS FAILED [pid={pipeline_id}]")
    return None


async def check_layer_health(layer_num: int) -> dict:
    urls = LAYER_URLS[layer_num]
    for url in urls:
        try:
            r = await state["http"].get(f"{url}/health", timeout=3.0)
            if r.status_code == 200:
                return {"online": True, "active_url": url}
        except:
            continue
    return {"online": False, "active_url": None}


@app.get("/health")
async def health():
    layer_status = {}
    for num in LAYER_URLS:
        status = await check_layer_health(num)
        layer_status[f"layer{num}"] = status["online"]
    return {
        "status":      "ok",
        "service":     "orchestrator",
        "port":        8000,
        "layers":      layer_status,
        "all_online":  all(layer_status.values()),
        "active_urls": _active_url,
    }


# ── FIX 9: Metrics endpoint for production monitoring ─────────────────────────
@app.get("/metrics")
async def get_metrics():
    """Get orchestrator metrics for monitoring."""
    return {
        "metrics": metrics.get_summary(),
        "rate_limiter": rate_limiter.get_stats(),
        "circuit_breaker": circuit_breaker.get_status(),
    }


@app.get("/mesh/status")
async def mesh_status():
    """Show full mesh status — which URL is active per layer and all fallbacks."""
    result = {}
    for num, urls in LAYER_URLS.items():
        layer_result = []
        for url in urls:
            try:
                r = await state["http"].get(f"{url}/health", timeout=2.0)
                online = r.status_code == 200
            except:
                online = False
            layer_result.append({"url": url, "online": online,
                                  "active": url == _active_url[num]})
        result[f"layer{num}"] = layer_result
    return result

# ── Relevance filter ──────────────────────────────────────────────────────────
_seen_alerts: dict = {}
_DEDUP_WINDOW = 60

def relevance_filter(l1: dict, alert) -> dict:
    score  = l1.get("anomaly_score", 0)
    conf   = l1.get("confidence", 0)
    method = l1.get("detection_method", "")
    src_ip = l1.get("source_ip", "")
    attack = l1.get("attack_type", "")

    if score < 0.3:
        return {"pass": False, "reason": "score_below_threshold",
                "quality": "noise", "retrain_eligible": False, "playbook_eligible": False}

    dedup_key = hashlib.md5(f"{src_ip}:{attack}".encode()).hexdigest()
    now = _time.time()
    if dedup_key in _seen_alerts:
        if now - _seen_alerts[dedup_key] < _DEDUP_WINDOW:
            return {"pass": False, "reason": "duplicate_within_60s",
                    "quality": "duplicate", "retrain_eligible": False, "playbook_eligible": False}
    _seen_alerts[dedup_key] = now

    if "novel_flag" in method or attack == "Unknown_Novel_Attack":
        return {"pass": True, "reason": "novel_attack_detected",
                "quality": "novel", "retrain_eligible": False, "playbook_eligible": True}

    if score >= 0.7 and conf >= 0.5:
        return {"pass": True, "reason": "high_confidence_detection",
                "quality": "high", "retrain_eligible": True, "playbook_eligible": True}

    if score >= 0.5:
        return {"pass": True, "reason": "statistical_detection",
                "quality": "medium", "retrain_eligible": False, "playbook_eligible": True}

    return {"pass": True, "reason": "marginal_signal",
            "quality": "low", "retrain_eligible": False, "playbook_eligible": False}

# ── Pipeline ──────────────────────────────────────────────────────────────────
@app.post("/pipeline/run")
async def run_pipeline(alert: Alert):
    # FIX 9: Rate limiting
    if not rate_limiter.acquire():
        metrics.inc("pipeline_rate_limited")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    pipeline_id = str(uuid.uuid4())
    started_at  = datetime.utcnow().isoformat()
    pipeline_start = _time.time()
    
    metrics.inc("pipeline_started")
    logger.info(f"Pipeline START [pid={pipeline_id}] alert={alert.alert_id}")
    
    result = {
        "pipeline_id":  pipeline_id,
        "alert_id":     alert.alert_id,
        "started_at":   started_at,
        "layer1":       None,
        "layer2":       None,
        "layer3":       None,
        "layer4":       None,
        "layer5":       None,
        "final_action": None,
        "playbook":     None,
    }

    # Layer 1
    l1 = await call_layer(1, "/detect", alert.model_dump(), pipeline_id)
    result["layer1"] = l1
    if not l1:
        result["error"] = "Layer 1 offline"
        metrics.inc("pipeline_l1_failed")
        return result

    if not l1.get("is_anomalous", False):
        result["verdict"]       = "BENIGN"
        result["anomaly_score"] = l1.get("anomaly_score", 0)
        metrics.inc("pipeline_benign")
        _log_to_es(result, alert)
        return result

    result["verdict"]       = "ANOMALOUS"
    result["anomaly_score"] = l1.get("anomaly_score", 0)
    metrics.inc("pipeline_anomalous")

    rf = relevance_filter(l1, alert)
    if not rf.get("pass"):
        result["verdict"] = "FILTERED"
        result["reason"]  = rf.get("reason")
        result["quality"] = rf.get("quality")
        metrics.inc(f"pipeline_filtered_{rf.get('reason', 'unknown')}")
        _log_to_es(result, alert)
        return result

    # L2 + L4 in parallel
    l2_payload = {
        "alert_id":       l1["alert_id"],
        "timestamp":      l1.get("timestamp", ""),
        "source_ip":      l1.get("source_ip", ""),
        "dest_ip":        l1.get("dest_ip", ""),
        "attack_type":    l1.get("attack_type", ""),
        "anomaly_score":  l1["anomaly_score"],
        "feature_vector": l1["embedding"],
    }
    # FIX 6: Layer 4 now uses 77 features (CICIDS format) - use cicids_features if available
    # Falls back to embedding[:77] padded to 77 dims if cicids_features not present
    _l4_feats = l1.get("cicids_features") or l1["embedding"][:77]
    if len(_l4_feats) < 77:
        _l4_feats = list(_l4_feats) + [0.0] * (77 - len(_l4_feats))
    l4_payload = {"features": _l4_feats[:77]}

    l2, l4 = await asyncio.gather(
        call_layer(2, "/correlate", l2_payload, pipeline_id),
        call_layer(4, "/predict",   l4_payload, pipeline_id),
    )
    result["layer2"] = l2
    result["layer4"] = l4

    if l2:
        l3_payload = {
            "alert_id":          l1["alert_id"],
            "timestamp":         l1.get("timestamp", ""),
            "source_ip":         l1.get("source_ip", ""),
            "destination_ip":    l1.get("dest_ip", ""),
            "dest_ip":           l1.get("dest_ip", ""),
            "source_port":       0,
            "destination_port":  0,
            "protocol":          "TCP",
            "severity":          "critical" if l1["anomaly_score"] > 0.7 else "high",
            "attack_type":       l1.get("attack_type", ""),
            "feature_vector":    l1["embedding"],
            "layer2_confidence": l2.get("confidence", 0.5),
        }
        l3 = await call_layer(3, "/respond", l3_payload, pipeline_id)
        result["layer3"] = l3
        if l3:
            if not l3.get("action"):
                l3["action"] = l3.get("primary_action") or \
                               (l3.get("actions") or [None])[0] or "monitor"
            result["final_action"] = l3
        else:
            result["final_action"] = {
                "chain_id": pipeline_id, "action": "monitor",
                "target_ip": alert.source_ip or "", "verified_safe": True,
                "q_value": 0.5, "fallback": True,
            }
    else:
        result["layer3"]       = None
        result["final_action"] = {
            "chain_id": pipeline_id, "action": "monitor",
            "target_ip": alert.source_ip or "", "verified_safe": True,
            "q_value": 0.5, "fallback": True,
        }

    # L4 retrain gate
    if rf.get("retrain_eligible") and l4 and l4.get("success"):
        try:
            # FIX 6: L4 retrain expects attack_features (list of 77-dim vectors) and attack_labels
            await call_layer(4, "/retrain", {
                "attack_features": [_l4_feats],
                "attack_labels":   [1],  # 1 = attack
                "trigger_source":  f"orchestrator_{l1.get('attack_type', 'unknown')}",
            }, pipeline_id)
        except Exception as e:
            logger.warning(f"L4 retrain skipped: {e} [pid={pipeline_id}]")

    # L5 playbook gate
    if not rf.get("playbook_eligible"):
        result["layer5"]   = None
        result["playbook"] = {
            "incident_id": pipeline_id,
            "attack_summary": "Low quality signal — playbook suppressed",
            "steps": [], "predicted_next": "unknown", "confidence": 0.0,
        }
        metrics.inc("pipeline_playbook_suppressed")
        _log_to_es(result, alert)
        return result

    l5_payload = result["final_action"]
    l5 = await call_layer(5, "/explain", l5_payload, pipeline_id)
    if l5 and not l5.get("predicted_next") and l5.get("predicted_threats"):
        l5["predicted_next"] = l5["predicted_threats"][0]
    result["layer5"] = l5

    if not l5:
        try:
            ollama_r = await state["http"].post(
                f"{LAYER_URLS[1][0]}/generate-playbook",
                json={
                    "prompt": "Generate a 3-step incident response playbook.",
                    "context": {
                        "attack_type":   l1.get("attack_type"),
                        "source_ip":     l1.get("source_ip"),
                        "anomaly_score": l1.get("anomaly_score"),
                        "method":        l1.get("detection_method"),
                    }
                },
                timeout=60.0
            )
            playbook_text = ollama_r.json().get("response", "")
            result["playbook"] = {
                "incident_id":    pipeline_id,
                "attack_summary": f"{l1.get('attack_type')} from {l1.get('source_ip')}",
                "steps":          [playbook_text],
                "predicted_next": "unknown",
                "confidence":     0.5,
                "source":         "layer1_ollama_fallback",
            }
            metrics.inc("pipeline_playbook_fallback")
        except Exception as e:
            logger.error(f"Playbook fallback failed: {e} [pid={pipeline_id}]")
            metrics.inc("pipeline_playbook_failed")
    else:
        result["playbook"] = l5
        metrics.inc("pipeline_playbook_l5")

    state["kafka"].send("playbooks", {
        "pipeline_id":   pipeline_id,
        "alert_id":      alert.alert_id,
        "verdict":       result["verdict"],
        "anomaly_score": result["anomaly_score"],
        "action":        (result.get("final_action") or {}).get("action", "unknown"),
        "timestamp":     started_at,
    })

    state["redis"].publish("immunex_pipeline", {
        "pipeline_id":   pipeline_id,
        "alert_id":      alert.alert_id,
        "verdict":       result["verdict"],
        "anomaly_score": result["anomaly_score"],
        "layers_online": {
            "l1": l1 is not None, "l2": l2 is not None,
            "l3": result["layer3"] is not None,
            "l4": l4 is not None, "l5": l5 is not None,
        }
    })

    _log_to_es(result, alert)
    
    # FIX 9: Final metrics and logging
    pipeline_duration_ms = (_time.time() - pipeline_start) * 1000
    metrics.inc("pipeline_completed")
    metrics.observe("pipeline_latency_ms", pipeline_duration_ms)
    
    result["completed_at"] = datetime.utcnow().isoformat()
    result["duration_ms"] = round(pipeline_duration_ms, 2)
    
    logger.info(
        f"Pipeline COMPLETE [pid={pipeline_id}] "
        f"verdict={result['verdict']} action={result.get('final_action', {}).get('action', 'n/a')} "
        f"duration={pipeline_duration_ms:.0f}ms"
    )
    return result

@app.post("/demo/inject")
async def demo_inject():
    botnet_features = [-0.7646905574813246,1.8493171336719367,0.1360842597127493,-0.038256832829822,-0.2234762935258288,0.1867425617760749,-0.2887230032189737,-0.3338156024101207,-0.3094501517292162,-0.243075338744169,4.544640932295218,-0.6642257447173439,3.1770112733598714,4.633082888390731,-0.1770146168201649,-0.1587662528427847,1.1936840927012191,2.3922403065334,2.765064551509225,-0.0582008046198575,1.862034595775524,0.8855807278018033,2.6763375243600733,2.762514242871944,-0.1243816297992087,-0.3613781146589914,-0.2127801830691396,-0.2475923127532377,-0.2842236442106687,-0.122538486443635,-0.1890309835329326,0.0,0.0,0.0,0.0373291387560878,-0.0817210922126444,-0.1349697802794396,-0.205328179738589,-0.7746180708204594,4.332156577732468,2.060003453281091,3.636782569436395,3.947944798307136,-0.1742105834718708,-0.1890309835329326,0.0,-0.5689403351968044,1.6932962780256324,-0.2973405807608872,0.0,0.0,-1.1283890896284317,2.027611601794204,-0.3094501517292162,3.1770112733598714,0.0,0.0,0.0,0.0,0.0,0.0,0.1360842597127493,-0.2234762935258288,-0.038256832829822,0.1867425617760749,-0.4425389824504542,-0.2020681797073581,0.3430337318398583,-0.8167862912590615,0.4715044104583414,-0.1382837801439019,0.2060428820616336,0.5834133109605503,2.897013722170425,-0.1150629802833818,2.7933872433630125,2.946668070457503]
    demo_alert = Alert(
        alert_id="DEMO-" + uuid.uuid4().hex[:8].upper(),
        timestamp=datetime.utcnow().isoformat(),
        source_ip="203.0.113.99", dest_ip="192.168.10.50",
        alert_type="Zeus_Banking_Trojan", severity="critical",
        features=botnet_features,
        text="Zeus banking trojan detected port scan from 203.0.113.99",
        event_type="FILE_ACCESS", protocol="TCP", port=445,
        username="svc_account_03", process="powershell.exe",
        file="C:\\finance\\transactions_Q1.xlsx", privilege_level="admin",
    )
    result = await run_pipeline(demo_alert)
    result["demo"] = True
    return result

@app.get("/pipeline/status")
async def pipeline_status():
    layer_status = {}
    for num in LAYER_URLS:
        status = await check_layer_health(num)
        layer_status[num] = status["online"]
    online  = [n for n, up in layer_status.items() if up]
    offline = [n for n, up in layer_status.items() if not up]
    return {
        "online":          online,
        "offline":         offline,
        "ready":           len(online) == 5,
        "layer1_critical": layer_status.get(1, False),
        "active_urls":     _active_url,
    }

def _log_to_es(result: dict, alert: Alert):
    try:
        state["es"].index_incident({
            "pipeline_id":   result.get("pipeline_id"),
            "alert_id":      alert.alert_id,
            "source_ip":     alert.source_ip,
            "dest_ip":       alert.dest_ip,
            "attack_type":   alert.alert_type,
            "verdict":       result.get("verdict", "unknown"),
            "anomaly_score": result.get("anomaly_score", 0),
            "layers_ran":    [k for k in ["layer1","layer2","layer3","layer4","layer5"]
                              if result.get(k) is not None],
        })
    except Exception as e:
        logger.error(f"ES log failed: {e}")

