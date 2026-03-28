import json, logging
from typing import Optional, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = {
    "raw_alerts":      "immunex_raw_alerts",
    "anomaly_results": "immunex_anomaly_results",
    "attack_graphs":   "immunex_attack_graphs",
    "responses":       "immunex_responses",
    "playbooks":       "immunex_playbooks",
}

class IMMUNEXProducer:
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP):
        self.producer = None
        self._bootstrap = bootstrap_servers
        self._connect()

    def _connect(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                request_timeout_ms=5000,
                api_version_auto_timeout_ms=5000,
            )
            logger.info("Kafka producer connected")
        except Exception as e:
            logger.warning(f"Kafka unavailable — running without event streaming: {e}")
            self.producer = None

    def send(self, topic_key: str, message: dict, key: Optional[str] = None):
        if self.producer is None:
            self._connect()
        if self.producer is None:
            logger.warning(f"Kafka offline — dropped message to {topic_key}")
            return False
        topic = TOPICS.get(topic_key, topic_key)
        try:
            future = self.producer.send(topic, value=message, key=key)
            record = future.get(timeout=10)
            logger.info(f"Sent to {topic} partition {record.partition} offset {record.offset}")
            return True
        except Exception as e:
            logger.error(f"Kafka send failed: {e}")
            self.producer = None
            return False

    def close(self):
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
            except Exception:
                pass


class IMMUNEXConsumer:
    def __init__(self, topic_key: str, group_id: str, bootstrap_servers=KAFKA_BOOTSTRAP):
        topic = TOPICS.get(topic_key, topic_key)
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        logger.info(f"Kafka consumer subscribed to {topic}")

    def consume(self, callback: Callable, max_messages: int = None):
        count = 0
        for message in self.consumer:
            callback(message.value)
            count += 1
            if max_messages and count >= max_messages:
                break

    def close(self):
        self.consumer.close()
