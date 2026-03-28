import logging
from typing import Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError

logger = logging.getLogger(__name__)

ES_HOST = "http://localhost:9200"

class IMMUNEXElastic:
    def __init__(self, host=ES_HOST):
        self.es = None
        self._host = host
        self._connect()

    def _connect(self):
        try:
            self.es = Elasticsearch(self._host, request_timeout=5)
            self._create_indices()
            logger.info("Elasticsearch connected")
        except Exception as e:
            logger.warning(f"Elasticsearch unavailable — running without ES logging: {e}")
            self.es = None

    def _create_indices(self):
        if not self.es:
            return
        for index in ["immunex_alerts", "immunex_incidents", "immunex_playbooks"]:
            try:
                if not self.es.indices.exists(index=index):
                    self.es.indices.create(index=index)
            except Exception as e:
                logger.warning(f"Could not create index {index}: {e}")

    def _ensure_connected(self):
        if self.es is None:
            self._connect()
        return self.es is not None

    def index_alert(self, doc: dict):
        if not self._ensure_connected():
            return False
        try:
            self.es.index(index="immunex_alerts", document=doc)
            return True
        except Exception as e:
            logger.warning(f"ES index_alert failed: {e}")
            self.es = None
            return False

    def index_incident(self, doc: dict):
        if not self._ensure_connected():
            return False
        try:
            self.es.index(index="immunex_incidents", document=doc)
            return True
        except Exception as e:
            logger.warning(f"ES index_incident failed: {e}")
            self.es = None
            return False

    def index_playbook(self, doc: dict):
        if not self._ensure_connected():
            return False
        try:
            self.es.index(index="immunex_playbooks", document=doc)
            return True
        except Exception as e:
            logger.warning(f"ES index_playbook failed: {e}")
            self.es = None
            return False

    def search(self, index: str, query: dict) -> list:
        if not self._ensure_connected():
            return []
        try:
            r = self.es.search(index=index, body=query)
            return [h["_source"] for h in r["hits"]["hits"]]
        except Exception as e:
            logger.warning(f"ES search failed: {e}")
            return []
