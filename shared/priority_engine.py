"""
shared/priority_engine.py
==========================
Priority scoring + async triage queue for IMMUNEX.

Scoring model (matches Splunk ES / QRadar risk scoring):
  final_score = (
      anomaly_score        * 0.35   # ML confidence from L1
    + severity_score       * 0.25   # raw severity field
    + asset_criticality    * 0.20   # how valuable is the target
    + attack_chain_score   * 0.15   # repeated attacks from same IP
    + mitre_tactic_score   * 0.05   # technique-based escalation
  ) * ioc_multiplier               # 2x if src_ip is known bad

Queue behaviour:
  - CRITICAL (score >= 0.85): immediate, bypasses queue
  - HIGH     (score >= 0.65): queue priority 1
  - MEDIUM   (score >= 0.40): queue priority 2  
  - LOW      (score <  0.40): queue priority 3, processed last
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logger = logging.getLogger("priority_engine")

# ── MITRE ATT&CK tactic → escalation weight ──────────────────────────────────
# Higher = more dangerous. Based on kill chain position.
MITRE_WEIGHTS = {
    # Initial access — moderate
    "PortScan":          0.40,
    "Reconnaissance":    0.35,
    "Phishing":          0.55,
    # Execution / persistence — serious
    "BruteForce":        0.60,
    "SQLi":              0.70,
    "CommandInjection":  0.75,
    "Backdoor":          0.80,
    # Privilege escalation / lateral movement — critical
    "PrivEsc":           0.85,
    "LateralMovement":   0.88,
    "C2":                0.90,
    # Exfiltration / impact — highest
    "Exfiltration":      0.95,
    "DataTheft":         0.95,
    "Ransomware":        1.00,
    "DDoS":              0.70,
    # Default for unknown
    "unknown":           0.45,
}

# ── Asset criticality map ─────────────────────────────────────────────────────
# In production this comes from a CMDB. Here we use subnet heuristics.
# Banks: 10.0.0.x = core banking, 10.0.1.x = trading, 10.0.2.x = DMZ, etc.
ASSET_CRITICALITY = {
    "10.0.0":  1.0,   # Core banking systems — maximum
    "10.0.1":  0.95,  # Trading systems
    "10.0.2":  0.70,  # DMZ / public-facing
    "10.0.3":  0.60,  # Internal services
    "10.0.4":  0.40,  # Dev/test
    "192.168": 0.50,  # Generic internal
    "172.16":  0.45,  # Internal range
}

def _asset_criticality(dest_ip: str) -> float:
    """Score how critical the target asset is."""
    for prefix, score in ASSET_CRITICALITY.items():
        if dest_ip.startswith(prefix):
            return score
    # External destination = data exfiltration risk
    if not dest_ip.startswith(("10.", "192.168.", "172.")):
        return 0.85  # outbound to external = high risk
    return 0.50  # unknown internal

def _severity_to_score(severity) -> float:
    """Normalize severity field to 0-1."""
    if isinstance(severity, float):
        return min(1.0, max(0.0, severity))
    mapping = {
        "critical": 1.0, "high": 0.75,
        "medium": 0.50,  "low": 0.25, "info": 0.10,
    }
    return mapping.get(str(severity).lower(), 0.5)

def _mitre_score(attack_type: str) -> float:
    """Map attack type to MITRE tactic weight."""
    for key, weight in MITRE_WEIGHTS.items():
        if key.lower() in attack_type.lower():
            return weight
    return MITRE_WEIGHTS["unknown"]


# ── Attack chain correlator ───────────────────────────────────────────────────
# Tracks how many times we've seen each src_ip in a rolling window.
# Repeated attacks from same IP = escalation (attacker is persistent).

class AttackChainTracker:
    def __init__(self, window_seconds: int = 300, escalation_threshold: int = 3):
        self.window = window_seconds
        self.threshold = escalation_threshold
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._raw_counts: dict[str, int] = defaultdict(int)  # pre-dedup counts

    def record_pre_dedup(self, src_ip: str, count: int = 1):
        """
        Called BEFORE Redis dedup with the raw count of how many times
        this IP appeared in the current batch. Tracks true attack volume
        even when dedup collapses duplicates to a single alert.
        """
        now = time.time()
        for _ in range(count):
            self._hits[src_ip].append(now)
        self._raw_counts[src_ip] += count

    def record_and_score(self, src_ip: str) -> float:
        """
        Record a hit from src_ip and return escalation score.
        Uses pre-dedup counts if available — dedup collapses
        repeated IPs to 1, so we must track raw volume separately.
        0.0 = first time seen, 1.0 = seen many times in window (persistent attacker).
        """
        now = time.time()
        # Prune old hits outside window
        self._hits[src_ip] = [t for t in self._hits[src_ip] if now - t < self.window]
        # Only append if not already recorded via record_pre_dedup
        if not self._hits[src_ip]:
            self._hits[src_ip].append(now)
        count = len(self._hits[src_ip])

        if count >= 10:
            return 1.0   # Sustained attack
        elif count >= 5:
            return 0.80  # Escalating
        elif count >= self.threshold:
            return 0.60  # Repeated
        elif count == 2:
            return 0.30  # Seen before
        return 0.0        # First time

    def get_hit_count(self, src_ip: str) -> int:
        now = time.time()
        return len([t for t in self._hits[src_ip] if now - t < self.window])

    def get_top_attackers(self, n: int = 10) -> list[tuple[str, int]]:
        """Return top N IPs by raw hit count in the current window."""
        now = time.time()
        window_counts = {
            ip: len([t for t in hits if now - t < self.window])
            for ip, hits in self._hits.items()
        }
        return sorted(window_counts.items(), key=lambda x: x[1], reverse=True)[:n]

# Global tracker instance
_chain_tracker = AttackChainTracker()


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_priority_score(l1_result: dict, original_alert: dict) -> dict:
    """
    Compute composite priority score for a detected anomaly.
    Returns enriched result with score, tier, and reasoning.
    
    Used by: orchestrator ingest_api after L1 returns anomalous results.
    """
    src_ip    = l1_result.get("source_ip", original_alert.get("source_ip", ""))
    dest_ip   = l1_result.get("dest_ip",   original_alert.get("dest_ip", ""))
    attack    = l1_result.get("attack_type", original_alert.get("alert_type", "unknown"))
    severity  = original_alert.get("severity", l1_result.get("confidence", 0.5))
    anomaly_score = l1_result.get("anomaly_score", 0.5)
    is_ioc    = original_alert.get("_ioc_hit", False)

    # Component scores
    s_anomaly   = float(anomaly_score)
    s_severity  = _severity_to_score(severity)
    s_asset     = _asset_criticality(dest_ip)
    s_chain     = _chain_tracker.record_and_score(src_ip)
    s_mitre     = _mitre_score(attack)

    # Weighted composite (weights sum to 1.0)
    raw_score = (
        s_anomaly  * 0.30 +
        s_severity * 0.20 +
        s_asset    * 0.20 +
        s_chain    * 0.20 +   # chain escalation now has real weight
        s_mitre    * 0.10
    )

    # IOC multiplier — known bad IPs get 2x (capped at 1.0)
    ioc_multiplier = 2.0 if is_ioc else 1.0
    final_score = min(1.0, raw_score * ioc_multiplier)

    # ── Hard overrides — certain combinations are always CRITICAL ────────────
    # These mirror what Splunk ES and QRadar do with correlation rules:
    # a high-confidence ML hit + critical severity + high-value target = no debate.
    is_critical_override = any([
        # Ransomware / exfil anywhere on the network
        s_mitre >= 0.90 and s_anomaly >= 0.70,
        # Any attack on core banking (10.0.0.x) with high ML confidence
        s_asset >= 0.95 and s_anomaly >= 0.75 and s_severity >= 0.70,
        # Persistent attacker (5+ hits) hitting a high-value target
        s_chain >= 0.80 and s_asset >= 0.60 and s_anomaly >= 0.65,
        # Known IOC on any target
        is_ioc and s_anomaly >= 0.60,
        # C2 or lateral movement regardless of asset
        s_mitre >= 0.85 and s_severity >= 0.70,
    ])

    if is_critical_override:
        final_score = max(final_score, 0.90)  # floor at 0.90 for overrides

    # Tier assignment
    if final_score >= 0.85:
        tier, queue_priority = "CRITICAL", 0
    elif final_score >= 0.65:
        tier, queue_priority = "HIGH", 1
    elif final_score >= 0.40:
        tier, queue_priority = "MEDIUM", 2
    else:
        tier, queue_priority = "LOW", 3

    hit_count = _chain_tracker.get_hit_count(src_ip)

    return {
        **l1_result,
        "priority_score":   round(final_score, 4),
        "priority_tier":    tier,
        "queue_priority":   queue_priority,
        "score_breakdown": {
            "anomaly":       round(s_anomaly, 3),
            "severity":      round(s_severity, 3),
            "asset_crit":    round(s_asset, 3),
            "attack_chain":  round(s_chain, 3),
            "mitre_tactic":  round(s_mitre, 3),
            "ioc_multiplier": ioc_multiplier,
        },
        "attack_chain_hits": hit_count,
        "mitre_tactic":      attack,
        "dest_criticality":  round(s_asset, 3),
        "escalated":         s_chain >= 0.60 or is_ioc,
    }


# ── Async Priority Queue ──────────────────────────────────────────────────────

@dataclass(order=True)
class PrioritizedAlert:
    """Wrapper for asyncio.PriorityQueue — lower priority_score = processed first."""
    queue_priority: int
    neg_score: float           # negative so highest score = lowest heap value
    timestamp: float = field(compare=False)
    alert: dict = field(compare=False)

    @classmethod
    def from_scored(cls, scored: dict) -> "PrioritizedAlert":
        return cls(
            queue_priority=scored["queue_priority"],
            neg_score=-scored["priority_score"],  # negate for max-heap behaviour
            timestamp=time.time(),
            alert=scored,
        )


class ThreatPriorityQueue:
    """
    Async priority queue for scored alerts.
    
    Usage:
        queue = ThreatPriorityQueue(maxsize=10000)
        await queue.put(scored_alert)
        alert = await queue.get()
    
    CRITICAL alerts bypass the queue entirely and are returned immediately.
    """

    def __init__(self, maxsize: int = 10000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        self._critical_buffer: list = []  # immediate processing, no queue
        self.stats = {
            "total_enqueued": 0,
            "critical_bypassed": 0,
            "dropped": 0,
        }

    async def put(self, scored_alert: dict):
        """Enqueue a scored alert. CRITICAL tier bypasses queue."""
        if scored_alert.get("priority_tier") == "CRITICAL":
            self._critical_buffer.append(scored_alert)
            self.stats["critical_bypassed"] += 1
            logger.warning(
                f"CRITICAL ALERT: {scored_alert.get('source_ip')} → "
                f"{scored_alert.get('dest_ip')} | "
                f"score={scored_alert.get('priority_score')} | "
                f"attack={scored_alert.get('attack_type')} | "
                f"chain_hits={scored_alert.get('attack_chain_hits')}"
            )
            return

        item = PrioritizedAlert.from_scored(scored_alert)
        try:
            self._queue.put_nowait(item)
            self.stats["total_enqueued"] += 1
        except asyncio.QueueFull:
            # Drop LOW priority if queue full, keep HIGH/MEDIUM
            if scored_alert.get("queue_priority", 3) < 3:
                await self._queue.put(item)  # block for HIGH/MEDIUM
                self.stats["total_enqueued"] += 1
            else:
                self.stats["dropped"] += 1
                logger.debug(f"Queue full, dropped LOW alert from {scored_alert.get('source_ip')}")

    def drain_critical(self) -> list[dict]:
        """Return and clear all CRITICAL alerts for immediate processing."""
        alerts = self._critical_buffer.copy()
        self._critical_buffer.clear()
        return alerts

    async def get(self) -> dict:
        """Get highest priority alert (blocks if empty)."""
        item = await self._queue.get()
        return item.alert

    def get_nowait(self) -> Optional[dict]:
        """Non-blocking get. Returns None if empty."""
        try:
            return self._queue.get_nowait().alert
        except asyncio.QueueEmpty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "queue_depth": self._queue.qsize(),
            "critical_pending": len(self._critical_buffer),
        }


# ── Global queue instance (used by orchestrator) ──────────────────────────────
_threat_queue: Optional[ThreatPriorityQueue] = None

def get_threat_queue() -> ThreatPriorityQueue:
    global _threat_queue
    if _threat_queue is None:
        _threat_queue = ThreatPriorityQueue(maxsize=50000)
    return _threat_queue


# ── Batch scoring helper ──────────────────────────────────────────────────────

async def score_and_enqueue_batch(
    anomalous_pairs: list[tuple[dict, dict]],
) -> tuple[list[dict], dict]:
    """
    Score all anomalous L1 results and enqueue by priority.
    Returns (scored_results, tier_counts).
    
    Call this after L1 detection, before L2-L5 pipeline.
    """
    queue = get_threat_queue()
    scored = []
    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for l1_result, original in anomalous_pairs:
        s = compute_priority_score(l1_result, original)
        scored.append(s)
        tier_counts[s["priority_tier"]] += 1
        await queue.put(s)

    # Sort by priority score descending for return value
    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    if tier_counts["CRITICAL"] > 0:
        logger.warning(
            f"TRIAGE: {tier_counts['CRITICAL']} CRITICAL | "
            f"{tier_counts['HIGH']} HIGH | "
            f"{tier_counts['MEDIUM']} MEDIUM | "
            f"{tier_counts['LOW']} LOW"
        )

    return scored, tier_counts
