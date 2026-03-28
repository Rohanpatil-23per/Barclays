import json, logging
from typing import Optional
import redis

logger = logging.getLogger(__name__)

REDIS_HOST = "localhost"
REDIS_PORT = 6379

class IMMUNEXCache:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.r.ping()
        logger.info("Redis connected")

    def cache_anomaly(self, alert_id: str, result: dict, ttl=3600):
        """Cache Layer 1 anomaly result for 1 hour."""
        self.r.setex(f"anomaly:{alert_id}", ttl, json.dumps(result))

    def get_anomaly(self, alert_id: str) -> Optional[dict]:
        val = self.r.get(f"anomaly:{alert_id}")
        return json.loads(val) if val else None

    def add_ioc(self, ip: str, threat_type: str, ttl=86400):
        """Add IP to IOC (Indicator of Compromise) list."""
        self.r.setex(f"ioc:{ip}", ttl, threat_type)

    def is_ioc(self, ip: str) -> Optional[str]:
        """Check if IP is a known IOC. Returns threat type or None."""
        return self.r.get(f"ioc:{ip}")

    def increment_alert_count(self, source_ip: str):
        """Track alert frequency per IP."""
        key = f"alert_count:{source_ip}"
        self.r.incr(key)
        self.r.expire(key, 3600)

    def get_alert_count(self, source_ip: str) -> int:
        return int(self.r.get(f"alert_count:{source_ip}") or 0)

    def cache_embedding(self, alert_id: str, embedding: list, ttl=7200):
        """Cache 768D RoBERTa embedding."""
        self.r.setex(f"emb:{alert_id}", ttl, json.dumps(embedding))

    def get_embedding(self, alert_id: str) -> Optional[list]:
        val = self.r.get(f"emb:{alert_id}")
        return json.loads(val) if val else None

    def publish(self, channel: str, message: dict):
        """Pub/sub for real-time dashboard updates."""
        self.r.publish(channel, json.dumps(message))