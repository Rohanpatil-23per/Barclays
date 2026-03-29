"""
layer1_detection/batch_endpoint.py  — FIXED
============================================
Two bugs fixed:
  1. _roberta_infer used norm-based score (always ~1.0) instead of softmax logits
     → replaced with correct classification head, same as single /detect
  2. Concurrent HTTP chunks all blocked on _gpu_lock → replaced with a single
     true batch call: all N alerts go through the GPU in one forward pass (chunked
     at 32 for VRAM safety), no per-chunk HTTP round-trips
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

_thread_pool = ThreadPoolExecutor(max_workers=2)   # 2 workers: iso+faiss can run alongside roberta
_faiss_lock  = threading.Lock()
_gpu_lock    = threading.Lock()   # one GPU forward pass at a time


class BatchAlert(BaseModel):
    alert_id:   str
    timestamp:  str
    source_ip:  str = "0.0.0.0"
    dest_ip:    str = "0.0.0.0"
    alert_type: str = "unknown"
    severity:   float = 0.5
    features:   list[float]

class BatchDetectRequest(BaseModel):
    alerts: list[BatchAlert]

class BatchDetectResponse(BaseModel):
    results:    list[dict]
    batch_size: int
    elapsed_ms: float
    throughput: float


# ── Fixed RoBERTa inference — uses softmax logits, same as single /detect ──────
def _roberta_infer_batch(model, tokenizer, texts: list[str], device: str):
    """
    True batched inference. Returns (attack_probs, embeddings) for all texts.
    attack_prob = softmax(logits)[1]  ← correct classification output
    This matches exactly what single /detect uses.
    """
    with _gpu_lock:
        model.eval()
        with torch.inference_mode():
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # Correct: softmax over logits — class 1 = attack probability
            probs      = torch.softmax(out.logits, dim=1)[:, 1].cpu().tolist()
            # Embedding from RoBERTa backbone CLS token
            embeddings = model.roberta(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"]
            ).last_hidden_state[:, 0, :].cpu().tolist()
        del out, enc
        if device == "cuda":
            torch.cuda.empty_cache()
    return probs, embeddings


def _iso_infer_batch(iso, scaler, features_array):
    scaled = scaler.transform(features_array)
    scores = iso.decision_function(scaled)
    # iso_flag = True means anomalous (score < threshold, usually -0.1)
    flags  = (scores < -0.1).tolist()
    return scores.tolist(), flags, scaled


def _faiss_search_batch(faiss_obj, emb_array):
    try:
        with _faiss_lock:
            D, _ = faiss_obj.index.search(emb_array.astype(np.float32), 5)
        threshold = getattr(faiss_obj, 'threshold', 0.7)
        return [(float(D[j].min()) > threshold, float(D[j].min()))
                for j in range(len(emb_array))]
    except Exception:
        return [(False, 0.0)] * len(emb_array)


# ── Core batch detection — all alerts in one GPU pass ─────────────────────────
async def _detect_batch_core(alerts: list[BatchAlert], state: dict) -> list[dict]:
    """
    Process all alerts in a single GPU pass (chunked at 32 for VRAM safety).
    No per-chunk HTTP round-trips. No concurrent GPU contention.
    """
    if not alerts:
        return []

    GPU_CHUNK = 32
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    loop      = asyncio.get_event_loop()

    # ── Redis cache check ──────────────────────────────────────────────────
    results_map = {}
    uncached    = []
    for i, alert in enumerate(alerts):
        try:
            cached = state["redis"].get_anomaly(alert.alert_id)
            if cached:
                results_map[i] = cached
                continue
        except Exception:
            pass
        uncached.append((i, alert))

    if not uncached:
        return [results_map[i] for i in range(len(alerts))]

    indices, batch = zip(*uncached)
    batch = list(batch)
    n     = len(batch)

    # ── Feature arrays ─────────────────────────────────────────────────────
    features_array = np.array(
        [a.features[:77] + [0.0] * max(0, 77 - len(a.features)) for a in batch],
        dtype=np.float32
    )

    # ── Isolation Forest (CPU, fast) ───────────────────────────────────────
    iso_scores, iso_flags, scaled = await loop.run_in_executor(
        _thread_pool, _iso_infer_batch,
        state["iso"], state["cicids_scaler"], features_array
    )

    # ── Build text sequences for RoBERTa ───────────────────────────────────
    from layer1_detection.inference_utils import load_top_features, serialize_features
    cicids_texts = [
        serialize_features(list(a.features[:77]), state["cicids_feats"])
        for a in batch
    ]

    unsw_array = state["unsw_scaler"].transform(features_array[:, :39])
    unsw_texts = [
        " ".join(
            f"{name.lower().replace(' ','_')}:{unsw_array[j][k]:.4f}"
            for k, name in enumerate(state["unsw_feats"][:25])
        )
        for j in range(n)
    ]

    # ── GPU inference — process in chunks of 32, sequential on single GPU ──
    # Sequential is correct here: concurrent forward passes on one GPU cause
    # CUDA memory errors. Chunking at 32 keeps VRAM usage under 2GB.
    all_cicids_probs  = []
    all_embeddings    = []
    all_unsw_probs    = []

    for chunk_start in range(0, n, GPU_CHUNK):
        chunk_end = min(chunk_start + GPU_CHUNK, n)

        c_probs, c_embs = await loop.run_in_executor(
            _thread_pool, _roberta_infer_batch,
            state["cicids_rob"], state["cicids_tok"],
            cicids_texts[chunk_start:chunk_end], device
        )
        u_probs, _ = await loop.run_in_executor(
            _thread_pool, _roberta_infer_batch,
            state["unsw_rob"], state["unsw_tok"],
            unsw_texts[chunk_start:chunk_end], device
        )

        all_cicids_probs.extend(c_probs)
        all_embeddings.extend(c_embs)
        all_unsw_probs.extend(u_probs)

    # ── FAISS batch search ─────────────────────────────────────────────────
    emb_array    = np.array(all_embeddings, dtype=np.float32)
    faiss_results = await loop.run_in_executor(
        _thread_pool, _faiss_search_batch, state["faiss"], emb_array
    )

    # ── IOC boosts ─────────────────────────────────────────────────────────
    ioc_boosts = []
    for a in batch:
        try:
            ioc_boosts.append(0.2 if state["redis"].is_ioc(a.source_ip) else 0.0)
        except Exception:
            ioc_boosts.append(0.0)

    # ── Decision fusion — same formula as single /detect ───────────────────
    new_results = {}
    for j, (orig_i, alert) in enumerate(zip(indices, batch)):
        attack_prob = min(1.0, 0.6 * all_cicids_probs[j] + 0.4 * all_unsw_probs[j] + ioc_boosts[j])
        iso_flag    = iso_flags[j]
        faiss_anom, _ = faiss_results[j]

        is_anomalous  = attack_prob > 0.5 or iso_flag or faiss_anom
        anomaly_score = float(np.clip(
            max(attack_prob, float(iso_flag) * 0.8, float(faiss_anom) * 0.7),
            0.0, 1.0
        ))

        if iso_flag and attack_prob > 0.5:   method = "both"
        elif iso_flag:                        method = "isolation_forest"
        elif faiss_anom:                      method = "faiss_similarity"
        else:                                 method = "ensemble"

        etype = alert.alert_type
        if attack_prob < 0.15 and anomaly_score > 0.6:
            etype  = "Unknown_Novel_Attack"
            method += "+novel_flag"

        result = {
            "alert_id":         alert.alert_id,
            "timestamp":        alert.timestamp,
            "source_ip":        alert.source_ip,
            "dest_ip":          alert.dest_ip,
            "attack_type":      etype,
            "anomaly_score":    round(anomaly_score, 4),
            "is_anomalous":     is_anomalous,
            "embedding":        all_embeddings[j],
            "detection_method": method,
            "confidence":       round(attack_prob, 4),
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

        t       = time.perf_counter()
        results = await _detect_batch_core(request.alerts, state)
        elapsed = (time.perf_counter() - t) * 1000
        n       = len(request.alerts)
        tps     = round(n / max(elapsed / 1000, 0.001), 1)
        anom    = sum(1 for r in results if r.get("is_anomalous"))
        benign  = n - anom

        logger.info(
            f"[BATCH] n={n} | {elapsed:.0f}ms | {tps} logs/s | "
            f"anomalous={anom} benign={benign} | "
            f"device={'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
        return BatchDetectResponse(
            results=results, batch_size=n,
            elapsed_ms=round(elapsed, 2), throughput=tps
        )

    logger.info("Registered: POST /detect/batch (true GPU batching, fixed scoring)")
    return detect_batch