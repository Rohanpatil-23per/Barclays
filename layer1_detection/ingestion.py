import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid, json, logging
from datetime import datetime
from typing import Optional
from shared.kafka_client import IMMUNEXProducer
from shared.redis_client import IMMUNEXCache
from shared.schemas import Alert

logger = logging.getLogger(__name__)

# Feature columns in exact order expected by Layer 1 models (77 features)
CICIDS_FEATURES = [
    "protocol","flow_duration","total_fwd_packets","total_backward_packets",
    "fwd_packets_length_total","bwd_packets_length_total","fwd_packet_length_max",
    "fwd_packet_length_min","fwd_packet_length_mean","fwd_packet_length_std",
    "bwd_packet_length_max","bwd_packet_length_min","bwd_packet_length_mean",
    "bwd_packet_length_std","flow_bytes_s","flow_packets_s","flow_iat_mean",
    "flow_iat_std","flow_iat_max","flow_iat_min","fwd_iat_total","fwd_iat_mean",
    "fwd_iat_std","fwd_iat_max","fwd_iat_min","bwd_iat_total","bwd_iat_mean",
    "bwd_iat_std","bwd_iat_max","bwd_iat_min","fwd_psh_flags","bwd_psh_flags",
    "fwd_urg_flags","bwd_urg_flags","fwd_header_length","bwd_header_length",
    "fwd_packets_s","bwd_packets_s","packet_length_min","packet_length_max",
    "packet_length_mean","packet_length_std","packet_length_variance",
    "fin_flag_count","syn_flag_count","rst_flag_count","psh_flag_count",
    "ack_flag_count","urg_flag_count","cwe_flag_count","ece_flag_count",
    "down_up_ratio","avg_packet_size","avg_fwd_segment_size","avg_bwd_segment_size",
    "fwd_avg_bytes_bulk","fwd_avg_packets_bulk","fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk","bwd_avg_packets_bulk","bwd_avg_bulk_rate",
    "subflow_fwd_packets","subflow_fwd_bytes","subflow_bwd_packets",
    "subflow_bwd_bytes","init_fwd_win_bytes","init_bwd_win_bytes",
    "fwd_act_data_packets","fwd_seg_size_min","active_mean","active_std",
    "active_max","active_min","idle_mean","idle_std","idle_max","idle_min"
]

# Source types IMMUNEX can ingest
SOURCE_TYPES = ["siem", "edr", "network", "auth", "firewall", "threat_feed"]


class EventNormalizer:
    """
    Normalizes raw events from SIEM/EDR/Network/Auth sources
    into unified Alert schema, then publishes to Kafka.
    """

    def __init__(self):
        self.producer = IMMUNEXProducer()
        self.cache    = IMMUNEXCache()
        logger.info("EventNormalizer initialized")

    def normalize(self, raw_event: dict, source_type: str) -> Optional[Alert]:
        """Convert raw event from any source to unified Alert schema."""
        try:
            if source_type == "siem":
                return self._from_siem(raw_event)
            elif source_type == "edr":
                return self._from_edr(raw_event)
            elif source_type == "network":
                return self._from_network(raw_event)
            elif source_type == "auth":
                return self._from_auth(raw_event)
            elif source_type == "firewall":
                return self._from_firewall(raw_event)
            elif source_type == "threat_feed":
                return self._from_threat_feed(raw_event)
            else:
                logger.warning(f"Unknown source type: {source_type}")
                return None
        except Exception as e:
            logger.error(f"Normalization failed for {source_type}: {e}")
            return None

    def ingest(self, raw_event: dict, source_type: str) -> bool:
        """Normalize and publish to Kafka immunex_raw_alerts topic."""
        alert = self.normalize(raw_event, source_type)
        if not alert:
            return False

        # Check IOC before publishing
        ioc_type = self.cache.is_ioc(alert.source_ip)
        if ioc_type:
            logger.warning(f"Known IOC detected: {alert.source_ip} ({ioc_type})")
            alert.severity = min(1.0, alert.severity + 0.3)

        # Track alert frequency
        self.cache.increment_alert_count(alert.source_ip)
        freq = self.cache.get_alert_count(alert.source_ip)
        if freq > 10:
            logger.warning(f"High frequency alerts from {alert.source_ip}: {freq}/hr")
            alert.severity = min(1.0, alert.severity + 0.2)

        # Publish to Kafka
        success = self.producer.send(
            "raw_alerts",
            alert.dict(),
            key=alert.alert_id
        )
        if success:
            logger.info(f"Ingested alert {alert.alert_id} from {source_type}")
        return success

    def _make_alert(self, source_ip, dest_ip, alert_type, severity, features=None):
        return Alert(
            alert_id   = str(uuid.uuid4()),
            timestamp  = datetime.utcnow().isoformat(),
            source_ip  = source_ip,
            dest_ip    = dest_ip,
            alert_type = alert_type,
            severity   = float(severity),
            features   = features or [0.0] * 77
        )

    def _from_siem(self, event: dict) -> Alert:
        """Normalize Syslog/CEF format from SIEM."""
        return self._make_alert(
            source_ip  = event.get("src_ip", event.get("source_ip", "0.0.0.0")),
            dest_ip    = event.get("dst_ip", event.get("dest_ip", "0.0.0.0")),
            alert_type = event.get("signature", event.get("alert_type", "unknown")),
            severity   = float(event.get("severity", 0.5)) / 10.0,
            features   = self._extract_features(event)
        )

    def _from_edr(self, event: dict) -> Alert:
        """Normalize EDR endpoint telemetry."""
        severity_map = {"critical": 0.95, "high": 0.75, "medium": 0.5, "low": 0.25}
        return self._make_alert(
            source_ip  = event.get("endpoint_ip", "0.0.0.0"),
            dest_ip    = event.get("remote_ip", "0.0.0.0"),
            alert_type = event.get("process_name", "edr_alert"),
            severity   = severity_map.get(event.get("severity", "medium"), 0.5),
            features   = self._extract_features(event)
        )

    def _from_network(self, event: dict) -> Alert:
        """Normalize network flow data (PCAP/NetFlow)."""
        return self._make_alert(
            source_ip  = event.get("src_ip", "0.0.0.0"),
            dest_ip    = event.get("dst_ip", "0.0.0.0"),
            alert_type = event.get("protocol", "network_flow"),
            severity   = min(1.0, float(event.get("bytes", 0)) / 1e6),
            features   = self._extract_features(event)
        )

    def _from_auth(self, event: dict) -> Alert:
        """Normalize authentication logs (LDAP/AD/Kerberos)."""
        failed = event.get("failed_attempts", 0)
        severity = min(1.0, failed / 20.0) if failed > 3 else 0.3
        return self._make_alert(
            source_ip  = event.get("client_ip", "0.0.0.0"),
            dest_ip    = event.get("server_ip", "0.0.0.0"),
            alert_type = "auth_failure" if failed > 3 else "auth_event",
            severity   = severity,
            features   = self._extract_features(event)
        )

    def _from_firewall(self, event: dict) -> Alert:
        """Normalize firewall logs."""
        action_severity = {"block": 0.7, "deny": 0.7, "drop": 0.6, "allow": 0.1}
        return self._make_alert(
            source_ip  = event.get("src_ip", "0.0.0.0"),
            dest_ip    = event.get("dst_ip", "0.0.0.0"),
            alert_type = f"firewall_{event.get('action', 'unknown')}",
            severity   = action_severity.get(event.get("action", "allow"), 0.3),
            features   = self._extract_features(event)
        )

    def _from_threat_feed(self, event: dict) -> Alert:
        """Normalize STIX/TAXII threat intelligence feed."""
        ip = event.get("indicator", "0.0.0.0")
        threat_type = event.get("threat_type", "unknown")
        self.cache.add_ioc(ip, threat_type, ttl=86400)
        return self._make_alert(
            source_ip  = ip,
            dest_ip    = "0.0.0.0",
            alert_type = threat_type,
            severity   = 0.9,
            features   = [0.0] * 77
        )

    def _extract_features(self, event: dict) -> list:
        """Extract 77 CICIDS features from event. Fills 0.0 for missing."""
        features = []
        for feat in CICIDS_FEATURES:
            val = event.get(feat, event.get(feat.replace("_", " "), 0.0))
            try:
                features.append(float(val))
            except (ValueError, TypeError):
                features.append(0.0)
        return features[:77]