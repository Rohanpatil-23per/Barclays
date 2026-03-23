import logging
from datetime import datetime
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

ES_HOST = "http://localhost:9200"

INDICES = {
    "logs":      "immunex_logs",
    "incidents": "immunex_incidents",
    "alerts":    "immunex_alerts",
}

MAPPINGS = {
    "mappings": {
        "properties": {
            "timestamp":     {"type": "date"},
            "alert_id":      {"type": "keyword"},
            "source_ip":     {"type": "ip"},
            "dest_ip":       {"type": "ip"},
            "attack_type":   {"type": "keyword"},
            "anomaly_score": {"type": "float"},
            "is_anomalous":  {"type": "boolean"},
            "layer":         {"type": "integer"},
            "raw":           {"type": "object", "enabled": False},
        }
    }
}

class IMMUNEXElastic:
    def __init__(self, host=ES_HOST):
        self.es = Elasticsearch(host)
        self._create_indices()
        logger.info("Elasticsearch connected")

    def _create_indices(self):
        for name, index in INDICES.items():
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, mappings=MAPPINGS["mappings"])
                logger.info(f"Created index: {index}")

    def index_alert(self, alert: dict, index_key="alerts"):
        index = INDICES.get(index_key, index_key)
        alert["@timestamp"] = datetime.utcnow().isoformat()
        self.es.index(index=index, document=alert)

    def index_incident(self, incident: dict):
        incident["@timestamp"] = datetime.utcnow().isoformat()
        self.es.index(index=INDICES["incidents"], document=incident)

    def search_similar(self, attack_type: str, limit=10):
        result = self.es.search(
            index=INDICES["incidents"],
            query={"match": {"attack_type": attack_type}},
            sort=[{"@timestamp": {"order": "desc", "unmapped_type": "date"}}],
            size=limit
        )
        return [h["_source"] for h in result["hits"]["hits"]]

    def get_recent(self, minutes=60, limit=100):
        result = self.es.search(
            index=INDICES["alerts"],
            query={"range": {"@timestamp": {"gte": f"now-{minutes}m"}}},
            sort=[{"@timestamp": {"order": "desc", "unmapped_type": "date"}}],
            size=limit
        )
        return [h["_source"] for h in result["hits"]["hits"]]