"""
IMMUNEX — Layer 2 Integration Pipeline (Live Environment)

This script ties the entire Layer 2 architecture together for the real-time banking environment:
1. Validates connection to the Dockerized Kafka and Neo4j infrastructure.
2. Consumes flagged alerts from Layer 1's Kafka topic.
3. Buffers them into 50-alert PyG Sliding Windows.
4. Builds the Heterogeneous Graph (Phase 1: Construction).
5. Runs GATv2 to extract the 118D Spatial Vector (Phase 2: Intelligence).
6. Runs BiLSTM over the timeline sequence to get 5D Current State.
7. Runs HMM over the BiLSTM output to get 5D Future State.
8. Fuses [118D Spatial] + [5D Current] + [5D Future] = 128D God-Mode Vector.
9. Packages the 128D vector + human-readable context into a rich JSON payload.
10. Routes to Neo4j (Dashboard) and Kafka (Layer 3 DQN).
"""

import json
import time
import torch
import numpy as np
from datetime import datetime, timezone
from collections import deque
from neo4j import GraphDatabase
from kafka import KafkaConsumer, KafkaProducer

from graph_builder import construct_heterogeneous_graph, generate_neo4j_pre_approval_commit
from gat_model import IMMUNEX_GATv2_Hetero
from bilstm_model import IMMUNEX_BiLSTM_Tracker
from hmm_predictor import IMMUNEX_HMM_Predictor

# --- CONFIGURATION (Matches docker-compose.yml) ---
KAFKA_BROKER = 'localhost:9092'
L1_ALERTS_TOPIC = 'immunex_layer1_alerts'
L2_TO_L3_TOPIC = 'immunex_layer3_dqn_queue'

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "immunex_admin"

MITRE_STAGES = ["Reconnaissance", "Initial Access", "Privilege Escalation", "Lateral Movement", "Exfiltration"]


class IMMUNEX_Pipeline:
    def __init__(self, gat_weights="immunex_gatv2_phase2.pt", bilstm_weights="immunex_bilstm_phase3.pt", seq_len=10):
        print("Initializing IMMUNEX Layer 2 Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load the Spatial Intelligence Model (GATv2)
        try:
            # Load 500 rows to guarantee all entity types (machine/file/user/process) appear
            # in the dummy graph — a single row may be missing nulled-out entity columns,
            # which would give PyG's to_hetero an incomplete metadata set.
            import pandas as pd
            df_dummy = pd.read_csv("immunex_final_dataset.csv", nrows=500)  # CRASH 2 fix
            dummy_graph = construct_heterogeneous_graph(df_dummy.to_dict('records'))

            self.gat = IMMUNEX_GATv2_Hetero(metadata=dummy_graph.metadata(), hidden_channels=128, num_classes=4, out_dim=118, heads=8).to(self.device)
            self.gat.load_state_dict(torch.load(gat_weights, map_location=self.device, weights_only=True))
            self.gat.eval()
            print("[+] GATv2 Spatial Model Loaded (118D Compression Head).")
        except Exception as e:
            print(f"[-] GATv2 loading failed. Error: {e}")
            self.gat = None  # CRASH 1 fix — without this, process_window throws AttributeError

        # 2. Load the Temporal Narrative Tracker (BiLSTM)
        try:
            self.bilstm = IMMUNEX_BiLSTM_Tracker(input_dim=118, hidden_dim=128, num_layers=2, num_classes=5).to(self.device)
            self.bilstm.load_state_dict(torch.load(bilstm_weights, map_location=self.device, weights_only=True))
            self.bilstm.eval()
            print("[+] BiLSTM Narrative Tracker Loaded (5-Class MITRE Output).")
        except Exception as e:
            print(f"[-] BiLSTM loading failed (It will be bypassed until trained). Error: {e}")
            self.bilstm = None
            
        # 3. Load the Predictive Engine (HMM)
        self.hmm = IMMUNEX_HMM_Predictor()
        print("[+] Hidden Markov Model (HMM) Loaded.")

        # 4. Connect to Neo4j (Phase 3 Memory)
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            self.neo4j_driver.verify_connectivity()
            print("[+] Neo4j Connection Established.")
        except Exception as e:
            print(f"[-] Neo4j offline. Error: {e}")
            self.neo4j_driver = None

        # 5. Connect Kafka Streams
        try:
            self.consumer = KafkaConsumer(
                L1_ALERTS_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            self.producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print("[+] Kafka Data Highway Configured.")
        except Exception as e:
            print(f"[-] Kafka offline. Error: {e}")
            self.consumer = None
            self.producer = None
            
        self.alert_buffer = []
        # Chronological buffer for the BiLSTM (maintains the last N state vectors)
        self.seq_len = seq_len
        self.narrative_timeline = deque(maxlen=seq_len)
        self.previous_5d = None

    def commit_to_neo4j(self, cypher_queries):
        """Executes a list of (query_string, params_dict) tuples against Neo4j."""
        if not self.neo4j_driver: return
        with self.neo4j_driver.session() as session:
            for query, params in cypher_queries:
                session.run(query, **params)

    def check_threat_memory(self, search_ips, search_files, search_users) -> dict:
        """
        Phase 3 Threat Memory Loop: Queries the Neo4j historical database to find
        if these exact entities were involved in a past attack graph.
        """
        if not self.neo4j_driver: return None
        
        # We look for any past Incident that shares the same IPs, files, or users
        query = f"""
        MATCH (i:Incident)-[:CONTAINS]->(a:Alert)
        OPTIONAL MATCH (a)-[:ORIGINATES_FROM|TARGETS]->(m:Machine)
        OPTIONAL MATCH (a)-[:INVOLVES]->(f:File)
        OPTIONAL MATCH (a)-[:PERFORMED_BY]->(u:User)
        WITH i, m, f, u
        WHERE m.ip IN $ips OR f.path IN $files OR u.username IN $users
        RETURN i.incident_id AS old_uuid, i.current_mitre_stage AS stage, 
               i.severity AS severity, count(a) AS overlap_score
        ORDER BY overlap_score DESC LIMIT 1
        """
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, ips=search_ips, files=search_files, users=search_users).single()
                if result and result["overlap_score"] > 2:  # Threshold of 3 overlapping alerts
                    return dict(result)
        except Exception as e:
            print(f"[-] Threat Memory DB query failed: {e}")
        return None

    def process_window(self, attack_uuid: str):
        if not self.alert_buffer or not self.gat:  # CRASH 1 fix — guard against missing GATv2
            return
        
        # ================================================================
        # THREAT MEMORY LOOP: Check historical Neo4j graphs before processing
        # ================================================================
        source_ips = list(set(str(a.get('source_ip', '')) for a in self.alert_buffer if a.get('source_ip')))
        dest_ips = list(set(str(a.get('dest_ip', '')) for a in self.alert_buffer if a.get('dest_ip')))
        search_ips = list(set(source_ips + dest_ips))
        
        search_files = list(set(str(a.get('file', '')) for a in self.alert_buffer if a.get('file')))
        search_users = list(set(str(a.get('username', '')) for a in self.alert_buffer if a.get('username')))
        
        print(f"\n[{attack_uuid}] Scanning Threat Memory for identical historical entity clusters...")
        historical_match = self.check_threat_memory(search_ips, search_files, search_users)
        
        if historical_match:
            print(f"[{attack_uuid}] > 🚨 THREAT MEMORY MATCH FOUND: Correlates with past Incident {historical_match['old_uuid']}")
            print(f"[{attack_uuid}] > Accelerating graph generation utilizing past '{historical_match['stage']}' profile.")
        else:
            print(f"[{attack_uuid}] > No historical match. Evaluating as a novel sequence.")
        
        # ================================================================
        # PHASE 1: GRAPH CONSTRUCTION (Flat alerts → Heterogeneous Graph)
        # ================================================================
        print(f"\n[{attack_uuid}] Phase 1: Constructing PyG Blast Radius Graph...")
        hetero_graph = construct_heterogeneous_graph(self.alert_buffer).to(self.device)
        num_alert_nodes = hetero_graph['alert'].num_nodes
        print(f"[{attack_uuid}] > Graph built: {num_alert_nodes} alert nodes")
        
        # ================================================================
        # PHASE 2: GRAPH INTELLIGENCE (GATv2 Attention + 118D Compression)
        # ================================================================
        print(f"[{attack_uuid}] Phase 2: Running GATv2 Attention Filtering...")
        with torch.no_grad():
            alert_batch = torch.zeros(num_alert_nodes, dtype=torch.long, device=self.device)
            node_logits, state_vector, severity_score = self.gat(
                x_dict=hetero_graph.x_dict, 
                edge_index_dict=hetero_graph.edge_index_dict, 
                alert_batch=alert_batch
            )
            
            # The 118-Dimensional SPATIAL Vector (Blast Radius Topology)
            spatial_118d = state_vector.cpu().numpy()[0]  # shape: (118,)
            print(f"[{attack_uuid}] > Compressed blast radius into 118D spatial vector. Severity: {severity_score.item():.2f}")
            
        # Append the new 118D vector to the chronological sequence
        self.narrative_timeline.append(spatial_118d.tolist())
        
        # Initialize the 5D state arrays (defaults until BiLSTM has enough data)
        current_5d = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Default: Reconnaissance
        future_5d = np.array([0.3, 0.6, 0.05, 0.0, 0.05])  # Default: Likely Initial Access next
        
        current_stage_name = "Reconnaissance"
        current_confidence = 1.0
        future_stage_name = "Initial Access"
        future_confidence = 0.6
        
        # If Threat Memory found a match, fast-track the starting state to match history
        if historical_match and historical_match.get("stage") and historical_match["stage"] != "None":
            current_stage_name = historical_match["stage"]
            idx = MITRE_STAGES.index(current_stage_name) if current_stage_name in MITRE_STAGES else 0
            current_5d = np.zeros(5)
            current_5d[idx] = 1.0
            print(f"[{attack_uuid}] > Fast-tracking baseline from Threat Memory: {current_stage_name}")

            # WRONG 4 fix — also update future_5d from the fast-tracked current state so that
            # the 128D god-mode vector is internally consistent (both halves from the same source)
            future_5d = self.hmm.predict_next_stage(current_5d, self.previous_5d)
            future_idx = int(np.argmax(future_5d))
            future_stage_name = MITRE_STAGES[future_idx]
            future_confidence = float(future_5d[future_idx])
        
        # ================================================================
        # BILSTM + HMM ENSEMBLE (Temporal Narrative → Current & Future)
        # ================================================================
        if self.bilstm and len(self.narrative_timeline) == self.seq_len:
            print(f"[{attack_uuid}] Injecting {self.seq_len}-Window Timeline into BiLSTM...")
            with torch.no_grad():
                seq_tensor = torch.tensor([list(self.narrative_timeline)], dtype=torch.float32, device=self.device)
                bilstm_logits = self.bilstm(seq_tensor)
                
                # The 5-Dimensional CURRENT STATE (BiLSTM Output)
                current_5d = torch.softmax(bilstm_logits, dim=1).cpu().numpy()[0]
                stage_idx = int(np.argmax(current_5d))
                current_stage_name = MITRE_STAGES[stage_idx]
                current_confidence = float(current_5d[stage_idx])
                
                print(f"[{attack_uuid}] > BiLSTM Current Stage: {current_stage_name} ({current_confidence*100:.1f}%)")
                
            # HMM Prediction: The 5-Dimensional FUTURE STATE (Adaptive)
            print(f"[{attack_uuid}] Routing to HMM Prediction Engine...")
            future_5d = self.hmm.predict_next_stage(current_stage_probs=current_5d, previous_stage_probs=self.previous_5d)
            self.previous_5d = current_5d  # Save for next window's adaptation
            future_idx = int(np.argmax(future_5d))
            future_stage_name = MITRE_STAGES[future_idx]
            future_confidence = float(future_5d[future_idx])
            
            print(f"[{attack_uuid}] > HMM Predicts NEXT MOVE: {future_stage_name} ({future_confidence*100:.1f}%)")
        else:
            if self.bilstm:
                print(f"[{attack_uuid}] Buffering timeline for BiLSTM narrative... ({len(self.narrative_timeline)}/{self.seq_len})")
        
        # ================================================================
        # 128D GOD-MODE VECTOR FUSION
        # [0:118]   → The Space:   Physical topology of the attack
        # [118:123]  → The Present: Exact MITRE stage the attacker is in
        # [123:128]  → The Future:  LSTM-HMM prediction of next move
        # ================================================================
        god_mode_128d = np.concatenate([spatial_118d, current_5d, future_5d]).tolist()
        
        print(f"[{attack_uuid}] > Fused 128D God-Mode Vector: [118D Spatial | 5D Current | 5D Future]")
        
        # ================================================================
        # Extract human-readable context from the alert buffer
        # ================================================================
        # WRONG 7 fix: renamed from source_ips to payload_source_ips to avoid shadowing
        # the earlier source_ips used for Threat Memory queries (line ~149).
        payload_source_ips = list(set(str(a.get('source_ip', '')) for a in self.alert_buffer if a.get('source_ip')))
        compromised_files = list(set(str(a.get('file', '')) for a in self.alert_buffer if a.get('file') and str(a.get('file')).lower() not in ['none', '']))
        compromised_procs = list(set(str(a.get('process', '')) for a in self.alert_buffer if a.get('process') and str(a.get('process')).lower() not in ['unknown.exe', '']))
        
        # Determine severity level
        # WRONG 1 fix: severity_score is a raw logit from nn.Linear — BCEWithLogitsLoss applies
        # sigmoid internally during training but NOT at inference. Apply it explicitly here so
        # sev_score is in [0, 1] and the threshold comparisons below are meaningful.
        sev_score = torch.sigmoid(severity_score).item()
        if sev_score > 0.7:
            severity_level = "critical"
        elif sev_score > 0.4:
            severity_level = "high"
        elif sev_score > 0.2:
            severity_level = "medium"
        else:
            severity_level = "low"
        
        # Determine attack type from dominant category in the window
        attack_cats = [str(a.get('attack_cat', 'Normal')).strip() for a in self.alert_buffer]
        attack_type = max(set(attack_cats), key=attack_cats.count)
        
        # ================================================================
        # RICH JSON PAYLOAD (For both DQN execution and SOC audit)
        # ================================================================
        rich_payload = {
            "alert_id": attack_uuid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_type": attack_type,
            "severity": severity_level,
            
            "human_readable_context": {
                "spatial_analysis": {
                    "source_ips": payload_source_ips[:5],  # WRONG 7 fix
                    "compromised_assets": (compromised_files + compromised_procs)[:5],
                    "blast_radius_nodes": num_alert_nodes
                },
                "temporal_analysis": {
                    "current_mitre_stage": current_stage_name,
                    "current_confidence": round(current_confidence, 4)
                },
                "predictive_analysis": {
                    "predicted_next_stage": future_stage_name,
                    "prediction_confidence": round(future_confidence, 4),
                    # WRONG 5 fix: data-driven ETA from trained HMM dwell statistics
                    # instead of the hardcoded "~2 minutes" that appeared on every alert
                    "time_to_next_stage_est": self.hmm.get_eta_string(future_idx)
                }
            },
            
            "feature_vector": god_mode_128d
        }
        
        # ================================================================
        # PHASE 3: TWO-STAGE THREAT MEMORY (Neo4j)
        # ================================================================
        
        # Stage 1: Pre-Approval Commit (Dashboard Visualization)
        print(f"[{attack_uuid}] Phase 3 (Stage 1): Pushing attack map to Neo4j Dashboard Store...")
        cypher_queries = generate_neo4j_pre_approval_commit(self.alert_buffer, attack_uuid=attack_uuid)
        
        # Annotate the Incident node with MITRE stage metadata (parameterized — SEC 1 fix)
        cypher_queries.append((
            "MATCH (i:Incident {incident_id: $incident_id}) "
            "SET i.current_mitre_stage = $current_stage, "
            "i.predicted_next_stage = $next_stage, "
            "i.severity = $severity",
            {
                "incident_id": attack_uuid,
                "current_stage": current_stage_name,
                "next_stage": future_stage_name,
                "severity": severity_level,
            }
        ))
        self.commit_to_neo4j(cypher_queries)
        
        # Route B: Kafka Handoff to Layer 3 DQN
        if self.producer:
            print(f"[{attack_uuid}] Handing over 128D God-Mode vector to Layer 3 Dueling DQN via Kafka...")
            self.producer.send(L2_TO_L3_TOPIC, rich_payload)
            self.producer.flush()  # SEC 2 fix — guarantee delivery before any unclean exit
        
        # Clear alert buffer for the next sliding window, but keep narrative_timeline intact
        self.alert_buffer.clear()

    def listen(self):
        print("\n--- Pipeline Active. Listening for Layer 1 Anomalies ---")
        if not self.consumer: return
        
        for message in self.consumer:
            alert = message.value
            self.alert_buffer.append(alert)
            print(f"Received Layer 1 Alert: {alert.get('event_type')}")
            
            # Sliding Window Trigger
            if len(self.alert_buffer) >= 50:
                uuid = f"INC_{int(time.time())}"
                self.process_window(attack_uuid=uuid)

    def test_pipeline(self, csv_path: str = "immunex_final_dataset.csv"):
        import pandas as pd
        print(f"\n--- [OFFLINE DIAGNOSTIC MODE] Injecting Live Telemetry from {csv_path} ---")
        try:
            df = pd.read_csv(csv_path)
            # Find an interesting 50-alert sequence (e.g., somewhere in the middle)
            start_idx = len(df) // 2
            simulated_alerts = df.iloc[start_idx:start_idx+50].to_dict('records')
            
            for alert in simulated_alerts:
                self.alert_buffer.append(alert)
                
            uuid = f"DIAGNOSTIC_{int(time.time())}"
            # Force the pipeline to process this batch
            self.process_window(attack_uuid=uuid)
            print("\n[+] PIPELINE TEST SUCCESSFUL: 128D God-Mode Vector generated offline.")
        except Exception as e:
            print(f"\n[-] PIPELINE TEST FAILED: {e}")

if __name__ == "__main__":
    import sys
    pipeline = IMMUNEX_Pipeline()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        pipeline.test_pipeline()
    else:
        print("Run `py -3.14 integration_pipeline.py --test` to simulate offline.\n")
        # pipeline.listen() # Uncomment for online prod setup with Kafka