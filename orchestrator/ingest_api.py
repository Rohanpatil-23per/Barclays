"""
orchestrator/ingest_api.py
==========================
High-throughput ingest endpoint for IMMUNEX.
Target: 50k-100k logs/s via:
  - Parallel normalization across all CPU cores
  - GPU-batched L1 inference (batch_size=32 per worker)
  - Distributed fan-out to L1 on all WireGuard nodes
  - Redis deduplication (90%+ cache hit rate on repeat IPs)
  - Kafka-backed async pipeline for anomalous logs only

Mount this in orchestrator/server.py:
    from orchestrator.ingest_api import router as ingest_router
    app.include_router(ingest_router)
"""

import asyncio
import hashlib
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Union

import httpx
import ssl
from orchestrator.mtls import get_client_ssl_context
_mtls_ctx = get_client_ssl_context(node=1)
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Import normalizer (hot path, no models)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.normalizer import normalize_bulk, normalize
from shared.priority_engine import score_and_enqueue_batch, get_threat_queue, _chain_tracker as _attack_tracker

logger = logging.getLogger("ingest_api")

router = APIRouter(prefix="/ingest", tags=["ingest"])

# ── Distributed L1 nodes (WireGuard mesh) ────────────────────────────────────
# Each node runs layer1_detection/server.py with /detect/batch endpoint
L1_NODES = [
    os.getenv("L1_NODE_1", "http://localhost:8001"),        # Your 4070 (primary)
    os.getenv("L1_NODE_2", "http://10.0.0.2:8001"),         # Acer Nitro 4050
    os.getenv("L1_NODE_3", "http://10.0.0.3:8001"),         # Lenovo LOQ 3050
    os.getenv("L1_NODE_4", "http://10.0.0.4:8001"),         # HP Victus 2050
    os.getenv("L1_NODE_5", "http://10.0.0.5:8001"),         # HP Pavilion 1650
]

# Track which nodes are alive (updated by health checker)
_node_alive: dict[str, bool] = {url: ("localhost" in url or "127.0.0.1" in url) for url in L1_NODES}
_node_device: dict[str, str] = {url: "unknown" for url in L1_NODES}
_node_latency: dict[str, float] = {url: 0.0 for url in L1_NODES}

# Global semaphore: prevent GPU OOM across all concurrent requests
_GPU_SEMAPHORE: asyncio.Semaphore = None
_PIPELINE_SEMAPHORE: asyncio.Semaphore = None  # L2-L5 pipeline concurrency

# Process pool for CPU-bound normalization
_process_pool: ProcessPoolExecutor = None

# HTTP client (shared, keep-alive connections)
_http: httpx.AsyncClient = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _http


def init_ingest(app_state: dict):
    """Call this from orchestrator lifespan to initialize shared resources."""
    global _GPU_SEMAPHORE, _PIPELINE_SEMAPHORE, _process_pool
    # Max 8 concurrent GPU batches (prevents VRAM OOM on 8GB card)
    _GPU_SEMAPHORE = asyncio.Semaphore(8)
    # Max 20 concurrent L2-L5 pipelines (anomalous logs only, ~5-10% of volume)
    _PIPELINE_SEMAPHORE = asyncio.Semaphore(20)
    # CPU workers for normalization (use all cores - 2)
    import multiprocessing
    workers = max(1, multiprocessing.cpu_count() - 2)
    _process_pool = ProcessPoolExecutor(max_workers=workers)
    logger.info(f"Ingest API initialized: {workers} CPU workers, 8 GPU slots, 20 pipeline slots")
    # Start background node health checker
    asyncio.create_task(_node_health_loop(app_state))


# ── Request / Response models ─────────────────────────────────────────────────

class RawLog(BaseModel):
    """A single raw log in any format."""
    data: Union[str, dict]
    source_hint: str = "auto"  # "siem"|"edr"|"network"|"firewall"|"auth"|"auto"


class BulkIngestRequest(BaseModel):
    logs: list[Union[str, dict]]
    run_full_pipeline: bool = False  # If True, anomalous logs go through L2-L5
    batch_size: int = 32             # GPU batch size per L1 node
    deduplicate: bool = True         # Redis dedup by source_ip+alert_type


class BulkIngestResponse(BaseModel):
    request_id: str
    total_received: int
    normalized: int
    deduplicated: int
    sent_to_l1: int
    anomalous: int
    benign: int
    elapsed_ms: float
    throughput_logs_per_sec: float
    format_breakdown: dict
    errors: list[str]


# ── Redis deduplication ────────────────────────────────────────────────────────

def _dedup_key(alert: dict) -> str:
    """Hash source_ip + alert_type for dedup window."""
    raw = f"{alert.get('source_ip','')}{alert.get('alert_type','')}"
    return hashlib.md5(raw.encode()).hexdigest()


async def _deduplicate(alerts: list[dict], redis, window_s: int = 60) -> tuple[list[dict], int]:
    """
    Remove duplicates using Redis SETEX.
    Returns (unique_alerts, num_deduped).
    ~0.1ms per alert via pipeline.
    """
    if not redis:
        return alerts, 0
    unique = []
    deduped = 0
    try:
        pipe = redis.r.pipeline()
        keys = [f"dedup:{_dedup_key(a)}" for a in alerts]
        for k in keys:
            pipe.set(k, 1, nx=True, ex=window_s)
        results = pipe.execute()
        for alert, was_set in zip(alerts, results):
            if was_set:  # nx=True: only set if not exists → True means it's new
                unique.append(alert)
            else:
                deduped += 1
    except Exception as e:
        logger.warning(f"Redis dedup failed, proceeding without dedup: {e}")
        return alerts, 0
    return unique, deduped


# ── Distributed L1 fan-out ────────────────────────────────────────────────────

def _get_live_nodes() -> list[str]:
    """Return GPU nodes that are alive, sorted by latency. CPU nodes excluded."""
    live = [(url, lat) for url, lat in _node_latency.items()
            if _node_alive.get(url, False)
            and _node_device.get(url, "unknown") in ("cuda", "unknown")]
    live.sort(key=lambda x: (x[1] == 0.0, x[1]))
    if not live:
        # Fallback: use any live node including CPU
        live = [(url, lat) for url, lat in _node_latency.items()
                if _node_alive.get(url, False)]
        live.sort(key=lambda x: x[1])
    return [url for url, _ in live] or [L1_NODES[0]]


def _partition_alerts(alerts: list[dict], nodes: list[str]) -> list[list[dict]]:
    """
    Distribute alerts across nodes evenly.
    Simple round-robin partition — each node gets alerts[i::n].
    """
    n = len(nodes)
    return [alerts[i::n] for i in range(n)]


async def _l1_batch_detect(node_url: str, alerts: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Send alerts to one L1 node in batches of batch_size.
    Uses /detect/batch endpoint (added to L1 server below).
    Falls back to /detect (single) if batch endpoint unavailable.
    """
    if not alerts:
        return []

    results = []
    http = get_http()

    # Build all chunks first, then fire them ALL concurrently (no per-chunk semaphore)
    chunks = [alerts[i:i + batch_size] for i in range(0, len(alerts), batch_size)]

    async def _send_chunk(chunk):
        try:
            t0 = time.perf_counter()
            r = await http.post(
                f"{node_url}/detect/batch",
                json={"alerts": chunk},
                timeout=60.0,
            )
            r.raise_for_status()
            _node_latency[node_url] = (time.perf_counter() - t0) * 1000
            return r.json().get("results", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"{node_url} has no /detect/batch, falling back to single")
                return await _l1_single_fallback(node_url, chunk)
            else:
                logger.error(f"L1 batch error at {node_url}: {e}")
                _node_alive[node_url] = False
                return []
        except Exception as e:
            logger.error(f"L1 node {node_url} failed: {e}")
            _node_alive[node_url] = False
            return []

    # Fire all chunks to this node concurrently — GPU handles batching internally
    async with _GPU_SEMAPHORE:
        chunk_results = await asyncio.gather(*[_send_chunk(c) for c in chunks])

    for cr in chunk_results:
        results.extend(cr)

    return results


async def _l1_single_fallback(node_url: str, alerts: list[dict]) -> list[dict]:
    """Fallback: call /detect one-by-one (slow, only if batch endpoint missing)."""
    http = get_http()
    tasks = [
        http.post(f"{node_url}/detect", json=a, timeout=30.0)
        for a in alerts
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for r in responses:
        if isinstance(r, Exception):
            continue
        try:
            r.raise_for_status()
            results.append(r.json())
        except Exception:
            pass
    return results


async def _fan_out_to_l1(alerts: list[dict], batch_size: int) -> list[dict]:
    """
    Fan out alerts to all live L1 nodes in parallel.
    Each node gets a partition of the alerts.
    """
    nodes = _get_live_nodes()
    partitions = _partition_alerts(alerts, nodes)

    tasks = [
        _l1_batch_detect(node_url, partition, batch_size)
        for node_url, partition in zip(nodes, partitions)
        if partition
    ]
    node_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for r in node_results:
        if isinstance(r, list):
            all_results.extend(r)
        elif isinstance(r, Exception):
            logger.error(f"L1 node task failed: {r}")

    return all_results


# ── Full pipeline for anomalous logs ─────────────────────────────────────────

async def _run_pipeline_for_anomalous(
    l1_result: dict,
    original_alert: dict,
    app_state: dict,
) -> dict:
    """
    Run L2-L5 pipeline for a single anomalous log.
    Gated by _PIPELINE_SEMAPHORE to prevent overload.
    """
    from shared.schemas import Alert
    async with _PIPELINE_SEMAPHORE:
        try:
            alert_obj = Alert(**original_alert)
            # Import run_pipeline from orchestrator
            from orchestrator.server import run_pipeline
            result = await run_pipeline(alert_obj)
            return result
        except Exception as e:
            logger.error(f"Pipeline failed for {original_alert.get('alert_id')}: {e}")
            return {"error": str(e), "l1": l1_result}


# ── Main bulk ingest endpoint ─────────────────────────────────────────────────

@router.post("/batch", response_model=BulkIngestResponse)
async def bulk_ingest(
    request: BulkIngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accept 50k-100k logs in any format (JSON, CSV, syslog, CEF, key-value).
    
    Flow:
      1. Normalize all logs (pure Python, ~16µs/log, ~60k logs/s)
      2. Redis dedup (removes repeated alerts from same IP)
      3. Fan-out to distributed L1 nodes in GPU batches of 32
      4. Return verdict counts immediately
      5. Anomalous logs optionally run through L2-L5 in background
    
    For 100k logs: expect 2-5 seconds total.
    """
    from orchestrator.server import state  # shared app state

    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())[:8].upper()
    errors = []

    if not request.logs:
        raise HTTPException(status_code=400, detail="No logs provided")

    total = len(request.logs)
    logger.info(f"[{request_id}] Bulk ingest: {total} logs")

    # ── Step 1: Normalize (CPU-bound, uses all cores) ─────────────────────────
    t_norm = time.perf_counter()
    alerts, norm_stats = normalize_bulk(request.logs)
    normalized = len(alerts)
    t_norm_ms = (time.perf_counter() - t_norm) * 1000

    # Record raw IP frequencies BEFORE dedup — chain tracker needs true volume
    from collections import Counter
    ip_counts = Counter(a.get("source_ip", "") for a in alerts if a.get("source_ip"))
    for ip, cnt in ip_counts.items():
        _attack_tracker.record_pre_dedup(ip, cnt)

    logger.info(f"[{request_id}] Normalized {normalized}/{total} in {t_norm_ms:.0f}ms "
                f"({total / max(t_norm_ms/1000, 0.001):.0f} logs/s)")

    if not alerts:
        elapsed = (time.perf_counter() - t_start) * 1000
        return BulkIngestResponse(
            request_id=request_id, total_received=total,
            normalized=0, deduplicated=0, sent_to_l1=0,
            anomalous=0, benign=0, elapsed_ms=elapsed,
            throughput_logs_per_sec=0,
            format_breakdown=norm_stats.get("formats", {}),
            errors=["All logs failed normalization"],
        )

    # ── Step 2: Redis dedup ───────────────────────────────────────────────────
    deduped = 0
    if request.deduplicate and state.get("redis"):
        alerts, deduped = await _deduplicate(alerts, state["redis"])
        logger.info(f"[{request_id}] Dedup: removed {deduped}, {len(alerts)} unique remaining")

    sent_to_l1 = len(alerts)
    if sent_to_l1 == 0:
        elapsed = (time.perf_counter() - t_start) * 1000
        return BulkIngestResponse(
            request_id=request_id, total_received=total,
            normalized=normalized, deduplicated=deduped,
            sent_to_l1=0, anomalous=0, benign=normalized,
            elapsed_ms=elapsed,
            throughput_logs_per_sec=total / max(elapsed / 1000, 0.001),
            format_breakdown=norm_stats.get("formats", {}),
            errors=[],
        )

    # ── Step 3: Fan-out to distributed L1 nodes ───────────────────────────────
    t_l1 = time.perf_counter()
    l1_results = await _fan_out_to_l1(alerts, request.batch_size)
    t_l1_ms = (time.perf_counter() - t_l1) * 1000
    logger.info(f"[{request_id}] L1 detection: {len(l1_results)} results in {t_l1_ms:.0f}ms")

    # ── Step 4: Partition anomalous vs benign ─────────────────────────────────
    anomalous_pairs = []
    benign_count = 0
    alert_map = {a["alert_id"]: a for a in alerts}

    for l1_res in l1_results:
        if not isinstance(l1_res, dict):
            continue
        if l1_res.get("is_anomalous", False):
            alert_id = l1_res.get("alert_id", "")
            original = alert_map.get(alert_id, {})
            anomalous_pairs.append((l1_res, original))
        else:
            benign_count += 1
            # Cache benign result in Redis
            if state.get("redis"):
                try:
                    state["redis"].cache_anomaly(
                        l1_res.get("alert_id", ""),
                        l1_res,
                        ttl=60
                    )
                except Exception:
                    pass

    anomalous_count = len(anomalous_pairs)

    # ── Priority scoring + triage queue ──────────────────────────────────────
    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    if anomalous_pairs:
        scored_results, tier_counts = await score_and_enqueue_batch(anomalous_pairs)
        # Replace anomalous_pairs with scored versions for pipeline
        anomalous_pairs = [(s, alert_map.get(s["alert_id"], {})) for s in scored_results]

    logger.info(f"[{request_id}] Verdict: {anomalous_count} anomalous | "
                f"CRITICAL={tier_counts['CRITICAL']} HIGH={tier_counts['HIGH']} "
                f"MEDIUM={tier_counts['MEDIUM']} LOW={tier_counts['LOW']}")

    # ── Step 5: Background pipeline for anomalous logs ───────────────────────
    if request.run_full_pipeline and anomalous_pairs and state:
        background_tasks.add_task(
            _run_anomalous_batch_pipeline,
            anomalous_pairs,
            state,
            request_id,
        )

    # ── Step 6: Kafka publish summary ─────────────────────────────────────────
    if state.get("kafka"):
        try:
            state["kafka"].send("raw_alerts", {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "total": total,
                "normalized": normalized,
                "anomalous": anomalous_count,
                "benign": benign_count,
            })
        except Exception:
            pass

    elapsed = (time.perf_counter() - t_start) * 1000
    throughput = total / max(elapsed / 1000, 0.001)

    logger.info(f"[{request_id}] Complete: {elapsed:.0f}ms, {throughput:.0f} logs/s")

    return BulkIngestResponse(
        request_id=request_id,
        total_received=total,
        normalized=normalized,
        deduplicated=deduped,
        sent_to_l1=sent_to_l1,
        anomalous=anomalous_count,
        benign=benign_count,
        elapsed_ms=round(elapsed, 2),
        throughput_logs_per_sec=round(throughput, 0),
        format_breakdown=norm_stats.get("formats", {}),
        errors=errors,
    )


async def _run_anomalous_batch_pipeline(
    pairs: list[tuple[dict, dict]],
    state: dict,
    request_id: str,
):
    """Background task: run L2-L5 for anomalous logs, respecting semaphore."""
    logger.info(f"[{request_id}] Running pipeline for {len(pairs)} anomalous logs")
    tasks = [
        _run_pipeline_for_anomalous(l1_res, original, state)
        for l1_res, original in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
    logger.info(f"[{request_id}] Pipeline complete: {success}/{len(pairs)} succeeded")


# ── Streaming ingest (for real-time feeds) ────────────────────────────────────

@router.post("/stream")
async def stream_ingest(log: RawLog):
    """
    Single log ingest — for real-time streaming use cases.
    Normalizes and runs full pipeline. Returns result synchronously.
    Use /batch for bulk processing.
    """
    from orchestrator.server import state, run_pipeline
    from shared.schemas import Alert

    alert_dict = normalize(log.data)
    if not alert_dict:
        raise HTTPException(status_code=422, detail="Could not normalize log")

    try:
        alert_obj = Alert(**alert_dict)
        result = await run_pipeline(alert_obj)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health / stats ─────────────────────────────────────────────────────────────

@router.get("/nodes")
async def node_status():
    """Show status of all distributed L1 nodes."""
    http = get_http()
    status = {}
    for url in L1_NODES:
        try:
            r = await http.get(f"{url}/health", timeout=2.0)
            data = r.json()
            status[url] = {
                "online": True,
                "device": data.get("device", "unknown"),
                "gpu_eligible": data.get("device", "unknown") == "cuda",
                "latency_ms": round(_node_latency.get(url, 0), 1),
                "alive": _node_alive.get(url, False),
            }
        except Exception:
            status[url] = {"online": False, "alive": False}
            _node_alive[url] = False
    live = sum(1 for s in status.values() if s["online"])
    return {
        "nodes": status,
        "live_count": live,
        "total_count": len(L1_NODES),
    }


# ── Priority queue stats endpoint ────────────────────────────────────────────

@router.get("/queue/stats")
async def queue_stats():
    """Live triage queue depth and tier breakdown."""
    from shared.priority_engine import get_threat_queue, _chain_tracker
    q = get_threat_queue()
    stats = q.get_stats()
    # Top 10 most active attacker IPs (uses pre-dedup raw counts)
    top_attackers = _chain_tracker.get_top_attackers(n=10)
    return {
        "queue": stats,
        "top_attackers": [{"ip": ip, "hits_5min": cnt} for ip, cnt in top_attackers],
    }

# ── Background node health checker ────────────────────────────────────────────

async def _node_health_loop(app_state: dict):
    """Ping all L1 nodes every 15s, update alive status."""
    http = get_http()
    while True:
        for url in L1_NODES:
            try:
                t0 = time.perf_counter()
                r = await http.get(f"{url}/health", timeout=2.0)
                data = r.json()
                _node_alive[url] = r.status_code == 200
                _node_device[url] = data.get("device", "unknown")
                _node_latency[url] = (time.perf_counter() - t0) * 1000
            except Exception:
                _node_alive[url] = False
        await asyncio.sleep(15)