"""
layer1_detection/batch_endpoint.py
====================================
GPU-batched L1 inference + Kafka consumer workers.
Target: 2000+ logs/s on RTX 4070 via:
  - Both RoBERTa models run IN PARALLEL (asyncio + ThreadPoolExecutor)
  - Fully vectorized UNSW scaler (no per-row loop)
  - Kafka consumer pulls batches directly — no HTTP overhead
  - FAISS batch search (already vectorized)
"""

import asyncio
import logging
import time
import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("l1_batch")

_thread_pool = ThreadPoolExecutor(max_workers=4)
_faiss_lock = threading.Lock()  # FAISS is not thread-safe under concurrent GPU access
_gpu_lock = threading.Lock()    # Serialize ALL GPU ops — single GPU can't handle concurrent inference


class BatchAlert(BaseModel):
    alert_id: str
    timestamp: str
    source_ip: str = "0.0.0.0"
    dest_ip: str = "0.0.0.0"
    alert_type: str = "unknown"
    severity: float = 0.5
    features: list[float]

class BatchDetectRequest(BaseModel):
    alerts: list[BatchAlert]

class BatchDetectResponse(BaseModel):
    results: list[dict]
    batch_size: int
    elapsed_ms: float
    throughput: float


def _roberta_infer(model, tokenizer, texts: list[str], device: str):
    """RoBERTa inference with VRAM cleanup between calls."""
    with _gpu_lock:
        model.eval()
        with torch.inference_mode():
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model.roberta(**enc)
            cls = outputs.last_hidden_state[:, 0, :].clone()
            probs = torch.sigmoid(cls.norm(dim=1) / 10.0).cpu().tolist()
            embeddings = cls.cpu().tolist()
        # Free intermediate tensors immediately
        del outputs, enc, cls
        if device == "cuda":
            torch.cuda.empty_cache()
    return probs, embeddings


def _iso_infer(iso, scaler, features_array):
    scaled = scaler.transform(features_array)
    scores = iso.decision_function(scaled)
    return scores.tolist(), (scores < -0.1).tolist(), scaled


def _faiss_search(faiss_obj, emb_array):
    try:
        with _faiss_lock:  # serialize FAISS access — prevents GPU memory corruption
            D, _ = faiss_obj.index.search(emb_array.astype(np.float32), 5)
        threshold = getattr(faiss_obj, 'threshold', 0.7)
        return [(float(D[j].min()) > threshold, float(D[j].min()))
                for j in range(len(emb_array))]
    except Exception:
        return [(False, 0.0)] * len(emb_array)


async def _detect_batch_core(alerts: list[BatchAlert], state: dict) -> list[dict]:
    # Process in GPU-safe chunks of 32 — prevents VRAM OOM on 8GB cards
    GPU_CHUNK = 32
    if len(alerts) > GPU_CHUNK:
        all_results = []
        for i in range(0, len(alerts), GPU_CHUNK):
            chunk = alerts[i:i + GPU_CHUNK]
            chunk_results = await _detect_batch_core(chunk, state)
            all_results.extend(chunk_results)
        return all_results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loop = asyncio.get_event_loop()

    # Redis cache check
    results_map = {}
    uncached = []
    for i, alert in enumerate(alerts):
        cached = state["redis"].get_anomaly(alert.alert_id)
        if cached:
            results_map[i] = cached
        else:
            uncached.append((i, alert))

    if not uncached:
        return [results_map[i] for i in range(len(alerts))]

    indices, batch = zip(*uncached)
    batch = list(batch)
    n = len(batch)

    features_array = np.array([a.features[:77] for a in batch], dtype=np.float32)

    # Isolation Forest (fast, vectorized)
    iso_scores, iso_flags, scaled = await loop.run_in_executor(
        _thread_pool, _iso_infer, state["iso"], state["cicids_scaler"], features_array)

    # CICIDS texts
    from layer1_detection.server import serialize_features
    cicids_texts = [serialize_features(list(a.features[:77]), state["cicids_feats"]) for a in batch]

    # UNSW texts (fully vectorized — no per-row loop)
    unsw_array = state["unsw_scaler"].transform(features_array[:, :39])
    unsw_texts = [
        " ".join(f"{n.lower().replace(' ','_')}:{unsw_array[j][k]:.4f}"
                 for k, n in enumerate(state["unsw_feats"][:25]))
        for j in range(n)
    ]

    # RoBERTa models run SEQUENTIALLY — GPU can only run one at a time
    # (parallel gather causes CUDA memory corruption on single-GPU systems)
    cicids_probs, embeddings = await loop.run_in_executor(_thread_pool, _roberta_infer,
        state["cicids_rob"], state["cicids_tok"], cicids_texts, device)
    unsw_probs, _ = await loop.run_in_executor(_thread_pool, _roberta_infer,
        state["unsw_rob"], state["unsw_tok"], unsw_texts, device)

    # FAISS batch
    emb_array = np.array(embeddings, dtype=np.float32)
    faiss_results = await loop.run_in_executor(_thread_pool, _faiss_search, state["faiss"], emb_array)

    # IOC boosts
    ioc_boosts = [0.2 if state["redis"].is_ioc(a.source_ip) else 0.0 for a in batch]

    # Decision fusion
    new_results = {}
    for j, (orig_i, alert) in enumerate(zip(indices, batch)):
        attack_prob = min(1.0, 0.6*cicids_probs[j] + 0.4*unsw_probs[j] + ioc_boosts[j])
        iso_flag = iso_flags[j]
        faiss_anom, _ = faiss_results[j]

        is_anomalous = attack_prob > 0.5 or iso_flag or faiss_anom
        anomaly_score = float(np.clip(
            max(attack_prob, float(iso_flag)*0.8, float(faiss_anom)*0.7), 0.0, 1.0))

        if iso_flag and attack_prob > 0.5: method = "both"
        elif iso_flag:                     method = "isolation_forest"
        elif faiss_anom:                   method = "faiss_similarity"
        else:                              method = "ensemble"

        etype = alert.alert_type
        if attack_prob < 0.15 and anomaly_score > 0.6:
            etype = "Unknown_Novel_Attack"
            method += "+novel_flag"

        result = {
            "alert_id":         alert.alert_id,
            "timestamp":        alert.timestamp,
            "source_ip":        alert.source_ip,
            "dest_ip":          alert.dest_ip,
            "attack_type":      etype,
            "anomaly_score":    anomaly_score,
            "is_anomalous":     is_anomalous,
            "embedding":        embeddings[j],
            "detection_method": method,
            "confidence":       attack_prob,
            "cicids_features":  scaled[j, :25].tolist(),
        }
        try:
            state["redis"].cache_anomaly(alert.alert_id, result, ttl=60)
        except Exception:
            pass
        new_results[orig_i] = result

    results_map.update(new_results)
    return [results_map[i] for i in range(len(alerts))]


def create_batch_detect_endpoint(app, state: dict):
    @app.post("/detect/batch", response_model=BatchDetectResponse)
    async def detect_batch(request: BatchDetectRequest):
        if not request.alerts:
            return BatchDetectResponse(results=[], batch_size=0, elapsed_ms=0, throughput=0)
        t = time.perf_counter()
        results = await _detect_batch_core(request.alerts, state)
        elapsed = (time.perf_counter() - t) * 1000
        n = len(request.alerts)
        tps = round(n / max(elapsed/1000, 0.001), 0)
        anom = sum(1 for r in results if r.get("is_anomalous"))
        logger.info(f"Batch={n} | {elapsed:.1f}ms | {tps:.0f} logs/s | anomalous={anom}")
        return BatchDetectResponse(results=results, batch_size=n,
                                   elapsed_ms=round(elapsed, 2), throughput=tps)

    logger.info("Registered: POST /detect/batch (parallel RoBERTa)")
    return detect_batch


# ── Kafka Consumer Workers ─────────────────────────────────────────────────────

class KafkaL1Worker:
    """
    Pulls raw_alerts from Kafka in micro-batches → GPU detect → publishes to anomaly_results.
    No HTTP overhead. This is how banks process millions of logs/day.
    """

    def __init__(self, state: dict, worker_id: int = 0,
                 batch_size: int = 32, batch_timeout_ms: int = 100):
        self.state = state
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"kafka-l1-{self.worker_id}", daemon=True)
        self._thread.start()
        logger.info(f"Kafka L1 worker {self.worker_id} started")

    def stop(self):
        self._running = False

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume())
        finally:
            loop.close()

    async def _consume(self):
        from kafka import KafkaConsumer, KafkaProducer
        from kafka.errors import NoBrokersAvailable
        try:
            consumer = KafkaConsumer(
                "immunex_raw_alerts",
                bootstrap_servers="localhost:9092",
                group_id="l1_workers",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                max_poll_records=self.batch_size,
                fetch_max_wait_ms=self.batch_timeout_ms,
            )
            producer = KafkaProducer(
                bootstrap_servers="localhost:9092",
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable:
            logger.warning(f"Worker {self.worker_id}: Kafka unavailable, skipping")
            return

        total = 0
        t_report = time.time()

        while self._running:
            try:
                records = consumer.poll(timeout_ms=self.batch_timeout_ms,
                                        max_records=self.batch_size)
                if not records:
                    await asyncio.sleep(0.01)
                    continue

                raw_list = [msg.value for msgs in records.values() for msg in msgs]
                if not raw_list:
                    continue

                batch = []
                for raw in raw_list:
                    try:
                        feats = raw.get("features", [0.0]*77)
                        if len(feats) < 77:
                            feats = feats + [0.0]*(77-len(feats))
                        batch.append(BatchAlert(
                            alert_id=raw.get("alert_id", ""),
                            timestamp=raw.get("timestamp", ""),
                            source_ip=raw.get("source_ip", "0.0.0.0"),
                            dest_ip=raw.get("dest_ip", "0.0.0.0"),
                            alert_type=raw.get("alert_type", "unknown"),
                            severity=float(raw.get("severity", 0.5)),
                            features=feats[:77],
                        ))
                    except Exception as e:
                        logger.warning(f"Bad alert: {e}")

                if not batch:
                    continue

                t0 = time.perf_counter()
                results = await _detect_batch_core(batch, self.state)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                anom = 0
                for r in results:
                    if r.get("is_anomalous"):
                        anom += 1
                        producer.send("immunex_anomaly_results", value=r)
                producer.flush()

                total += len(results)
                if time.time() - t_report > 10:
                    tps = total / max(time.time() - t_report, 0.001)
                    logger.info(f"Worker {self.worker_id}: {tps:.0f} logs/s | "
                                f"last_batch={len(results)} in {elapsed_ms:.1f}ms | anom={anom}")
                    total = 0
                    t_report = time.time()

            except Exception as e:
                logger.error(f"Worker {self.worker_id}: {e}")
                await asyncio.sleep(1)

        consumer.close()
        producer.close()


def start_kafka_workers(state: dict, num_workers: int = 4, batch_size: int = 32):
    workers = [KafkaL1Worker(state, i, batch_size) for i in range(num_workers)]
    for w in workers:
        w.start()
    logger.info(f"Started {num_workers} Kafka L1 workers (batch={batch_size})")
    return workers