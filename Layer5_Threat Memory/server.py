"""
IMMUNEX Layer 5 — Threat Memory & Prediction
FastAPI server on :8005

Models:
  - LSTM          : immunex_lstm_final.pt  — sequence-based attack prediction
  - CategoricalHMM: immunex_hmm.pkl        — hidden state (kill chain stage) inference

Observation mapping (from handoff doc):
  0=port_scan, 1=dns_query, 2=phishing_click, 3=login_fail, 4=login_success
  5=priv_escalation, 6=lateral_movement, 7=file_access, 8=large_upload, 9=zip_creation

HMM States: Recon → Init_Access → Priv_Esc → Lateral_Mv → Exfiltration

Orchestrator sends to POST /predict (via /explain in orchestrator):
  The final_action dict from L3, which includes chain_id, action, target_ip, etc.

Returns:
  {
    "chain_id":          str,
    "current_state":     str,   # HMM inferred kill chain stage
    "predicted_threats": list,  # next likely attack types
    "time_window":       str,   # "2-6 hours"
    "confidence":        float,
    "playbook":          str,   # LLM-style response recommendation
    "hmm_state_probs":   list,  # per-state probabilities
    "risk_level":        str    # HIGH / MEDIUM / LOW
  }
"""

import os
import pickle
import logging
import traceback
import sqlite3
import threading
import json
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [L5] %(levelname)s %(message)s"
)
logger = logging.getLogger("layer5")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "threat_memory.db")


# ─────────────────────────────────────────────────────────────
# SQLite Persistence Layer — FIX 8
# ─────────────────────────────────────────────────────────────
class ThreatMemoryDB:
    """
    Thread-safe SQLite persistence for Layer 5 Threat Memory.
    Stores:
      - Attack chains (chain_id → observations, state, metadata)
      - Prediction history for audit trail
      - Observation sequences per attacker IP
    """
    _local = threading.local()

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self):
        """Context manager for cursor with auto-commit."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _init_db(self):
        """Initialize database schema."""
        with self._cursor() as cur:
            # Attack chains table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attack_chains (
                    chain_id TEXT PRIMARY KEY,
                    target_ip TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    current_state TEXT DEFAULT 'Reconnaissance',
                    risk_level TEXT DEFAULT 'LOW',
                    observation_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Observations table (sequence per chain)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    obs_name TEXT NOT NULL,
                    obs_id INTEGER NOT NULL,
                    attack_type TEXT,
                    FOREIGN KEY (chain_id) REFERENCES attack_chains(chain_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_chain ON observations(chain_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp)")

            # Predictions audit log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_id TEXT,
                    timestamp TEXT NOT NULL,
                    target_ip TEXT,
                    attack_type TEXT,
                    predicted_state TEXT,
                    risk_level TEXT,
                    confidence REAL,
                    lstm_state TEXT,
                    hmm_state TEXT,
                    predicted_threats TEXT,
                    playbook TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_chain ON predictions(chain_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(timestamp)")

            # Attacker profiles (IP-based aggregation)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attacker_profiles (
                    ip TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_observations INTEGER DEFAULT 0,
                    max_state_reached TEXT DEFAULT 'Reconnaissance',
                    chain_ids TEXT DEFAULT '[]'
                )
            """)

        logger.info(f"SQLite DB initialized at {self.db_path}")

    def get_or_create_chain(self, chain_id: str, target_ip: str) -> Dict:
        """Get existing chain or create new one."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("SELECT * FROM attack_chains WHERE chain_id = ?", (chain_id,))
            row = cur.fetchone()
            if row:
                return dict(row)

            # Create new chain
            cur.execute("""
                INSERT INTO attack_chains (chain_id, target_ip, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (chain_id, target_ip, now, now))
            return {
                "chain_id": chain_id,
                "target_ip": target_ip,
                "first_seen": now,
                "last_seen": now,
                "current_state": "Reconnaissance",
                "risk_level": "LOW",
                "observation_count": 0,
                "is_active": 1
            }

    def add_observation(self, chain_id: str, obs_name: str, obs_id: int, attack_type: str = None):
        """Add an observation to a chain's sequence."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO observations (chain_id, timestamp, obs_name, obs_id, attack_type)
                VALUES (?, ?, ?, ?, ?)
            """, (chain_id, now, obs_name, obs_id, attack_type))

            cur.execute("""
                UPDATE attack_chains 
                SET last_seen = ?, observation_count = observation_count + 1
                WHERE chain_id = ?
            """, (now, chain_id))

    def get_observation_sequence(self, chain_id: str, limit: int = 50) -> List[int]:
        """Get observation sequence for a chain (most recent first, then reversed)."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT obs_id FROM observations 
                WHERE chain_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (chain_id, limit))
            rows = cur.fetchall()
            return [r['obs_id'] for r in reversed(rows)]

    def update_chain_state(self, chain_id: str, state: str, risk_level: str):
        """Update chain's current state and risk level."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                UPDATE attack_chains 
                SET current_state = ?, risk_level = ?, last_seen = ?
                WHERE chain_id = ?
            """, (state, risk_level, now, chain_id))

    def log_prediction(self, prediction: Dict):
        """Log prediction for audit trail."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO predictions 
                (chain_id, timestamp, target_ip, attack_type, predicted_state, 
                 risk_level, confidence, lstm_state, hmm_state, predicted_threats, playbook)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.get("chain_id"),
                prediction.get("timestamp"),
                prediction.get("target_ip"),
                prediction.get("attack_type"),
                prediction.get("current_state"),
                prediction.get("risk_level"),
                prediction.get("confidence"),
                prediction.get("lstm_state"),
                prediction.get("hmm_state"),
                json.dumps(prediction.get("predicted_threats", [])),
                prediction.get("playbook")
            ))

    def update_attacker_profile(self, ip: str, chain_id: str, state: str):
        """Update or create attacker profile."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("SELECT * FROM attacker_profiles WHERE ip = ?", (ip,))
            row = cur.fetchone()

            if row:
                chain_ids = json.loads(row['chain_ids'])
                if chain_id not in chain_ids:
                    chain_ids.append(chain_id)
                # Track highest state reached
                state_order = ["Reconnaissance", "Initial_Access", "Privilege_Escalation", 
                              "Lateral_Movement", "Exfiltration"]
                current_max = row['max_state_reached']
                if state in state_order and current_max in state_order:
                    if state_order.index(state) > state_order.index(current_max):
                        current_max = state

                cur.execute("""
                    UPDATE attacker_profiles 
                    SET last_seen = ?, total_observations = total_observations + 1,
                        max_state_reached = ?, chain_ids = ?
                    WHERE ip = ?
                """, (now, current_max, json.dumps(chain_ids), ip))
            else:
                cur.execute("""
                    INSERT INTO attacker_profiles 
                    (ip, first_seen, last_seen, total_observations, max_state_reached, chain_ids)
                    VALUES (?, ?, ?, 1, ?, ?)
                """, (ip, now, now, state, json.dumps([chain_id])))

    def get_chain_history(self, chain_id: str) -> Dict:
        """Get full chain history for analysis."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM attack_chains WHERE chain_id = ?", (chain_id,))
            chain = cur.fetchone()
            if not chain:
                return None

            cur.execute("""
                SELECT * FROM observations WHERE chain_id = ?
                ORDER BY timestamp ASC
            """, (chain_id,))
            observations = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT * FROM predictions WHERE chain_id = ?
                ORDER BY timestamp DESC LIMIT 10
            """, (chain_id,))
            recent_predictions = [dict(r) for r in cur.fetchall()]

            return {
                "chain": dict(chain),
                "observations": observations,
                "recent_predictions": recent_predictions
            }

    def get_active_chains(self, limit: int = 100) -> List[Dict]:
        """Get active attack chains."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM attack_chains 
                WHERE is_active = 1
                ORDER BY last_seen DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_attacker_profile(self, ip: str) -> Dict:
        """Get attacker profile by IP."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM attacker_profiles WHERE ip = ?", (ip,))
            row = cur.fetchone()
            if row:
                profile = dict(row)
                profile['chain_ids'] = json.loads(profile['chain_ids'])
                return profile
            return None

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM attack_chains WHERE is_active = 1")
            active_chains = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM observations")
            total_obs = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM predictions")
            total_preds = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM attacker_profiles")
            total_attackers = cur.fetchone()['cnt']

            cur.execute("""
                SELECT risk_level, COUNT(*) as cnt 
                FROM attack_chains WHERE is_active = 1
                GROUP BY risk_level
            """)
            risk_dist = {r['risk_level']: r['cnt'] for r in cur.fetchall()}

            return {
                "active_chains": active_chains,
                "total_observations": total_obs,
                "total_predictions": total_preds,
                "known_attackers": total_attackers,
                "risk_distribution": risk_dist
            }


# Global DB instance
threat_db = ThreatMemoryDB()


# ─────────────────────────────────────────────────────────────
# Observation + State mappings
# ─────────────────────────────────────────────────────────────
OBS_TO_ID = {
    "port_scan":       0, "dns_query":        1,
    "phishing_click":  2, "login_fail":       3,
    "login_success":   4, "priv_escalation":  5,
    "lateral_movement":6, "file_access":      7,
    "large_upload":    8, "zip_creation":     9,
}

# Map attack_type from L3 → observation
ATTACK_TO_OBS = {
    "PortScan":                  "port_scan",
    "Reconnaissance":            "port_scan",
    "Fuzzers":                   "port_scan",
    "FTP-Patator":               "login_fail",
    "SSH-Patator":               "login_fail",
    "Bot":                       "login_success",
    "Backdoor":                  "priv_escalation",
    "Worms":                     "lateral_movement",
    "DDoS":                      "large_upload",
    "DoS Hulk":                  "large_upload",
    "DoS GoldenEye":             "large_upload",
    "DoS slowloris":             "large_upload",
    "Infiltration":              "large_upload",
    "Heartbleed":                "priv_escalation",
    "Web Attack - Brute Force":  "login_fail",
    "Web Attack - XSS":          "file_access",
    "Web Attack - Sql Injection":"file_access",
    "Zeus Banking Trojan":       "lateral_movement",
    "Banking Trojan":            "lateral_movement",
}

HMM_STATES = [
    "Reconnaissance",
    "Initial_Access",
    "Privilege_Escalation",
    "Lateral_Movement",
    "Exfiltration",
]

# Next predicted threats per HMM state
STATE_TO_THREATS = {
    "Reconnaissance":        ["SSH brute force", "Credential stuffing", "Phishing"],
    "Initial_Access":        ["Privilege escalation", "Backdoor installation", "Lateral movement"],
    "Privilege_Escalation":  ["Lateral movement", "Credential dumping", "Persistence"],
    "Lateral_Movement":      ["Data staging", "Large file transfers", "C2 beaconing"],
    "Exfiltration":          ["Data destruction", "Ransomware deployment", "Cover tracks"],
}

STATE_TO_PLAYBOOK = {
    "Reconnaissance":       "Isolate scanning source IP. Enable enhanced logging on perimeter. Alert SOC tier 1.",
    "Initial_Access":       "Block attacker IP at firewall. Reset compromised credentials. Initiate forensic capture.",
    "Privilege_Escalation": "Revoke elevated privileges. Kill suspicious processes. Isolate affected host.",
    "Lateral_Movement":     "Segment affected VLAN. Block inter-host SMB/RDP. Initiate full IR playbook.",
    "Exfiltration":         "CRITICAL: Block all outbound to attacker C2. Preserve forensic evidence. Escalate to CISO.",
}

STATE_RISK = {
    "Reconnaissance":       "LOW",
    "Initial_Access":       "MEDIUM",
    "Privilege_Escalation": "HIGH",
    "Lateral_Movement":     "HIGH",
    "Exfiltration":         "CRITICAL",
}


# ─────────────────────────────────────────────────────────────
# LSTM model — matches training architecture exactly
# ─────────────────────────────────────────────────────────────
class LSTMAttackerPredictor(nn.Module):
    """
    LSTM model that predicts attacker stage from observation history.
    Architecture must match immunex_lstm_final.pt checkpoint exactly:
      - Embedding layer for observations
      - 2-layer LSTM
      - Two output heads: obs_head (next obs), state_head (current stage)
    """
    def __init__(self, n_obs=10, n_states=5, embed_dim=32, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(n_obs + 1, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout    = nn.Dropout(dropout)
        self.obs_head   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_obs)
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_states)
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, _   = self.lstm(embedded)
        last     = self.dropout(out[:, -1, :])
        return self.obs_head(last), self.state_head(last)


# ─────────────────────────────────────────────────────────────
# Model manager
# ─────────────────────────────────────────────────────────────
class L5Models:
    def __init__(self):
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm     = None
        self.hmm      = None
        self.lstm_ok  = False
        self.hmm_ok   = False
        self.n_hmm_states = 5
        self.seq_length = 19  # SEQ_LEN - 1 from training config

    def load(self, lstm_path: str, hmm_path: str):
        # ── LSTM ─────────────────────────────────────────────
        try:
            raw = torch.load(lstm_path, map_location="cpu", weights_only=False)
            state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw.state_dict()
            config = raw.get("config", {}) if isinstance(raw, dict) else {}

            # Extract config from checkpoint or use defaults matching training
            n_obs = config.get("n_observations", 10)
            n_states = config.get("n_states", 5)
            embed_dim = config.get("embed_dim", 32)
            hidden_dim = config.get("lstm_hidden", 128)
            n_layers = config.get("lstm_layers", 2)
            dropout = config.get("dropout", 0.3)
            self.seq_length = config.get("seq_length", 20) - 1

            logger.info(f"LSTM config: n_obs={n_obs}, n_states={n_states}, embed={embed_dim}, hidden={hidden_dim}, layers={n_layers}")

            self.lstm = LSTMAttackerPredictor(
                n_obs=n_obs,
                n_states=n_states,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout
            ).to(self.device)

            # Load with strict=True since architecture now matches
            self.lstm.load_state_dict(state_dict, strict=True)
            self.lstm.eval()
            self.lstm_ok = True
            logger.info(f"LSTM loaded with strict=True on {self.device}")
        except Exception as e:
            logger.warning(f"LSTM load failed: {e} — HMM-only mode")
            import traceback
            traceback.print_exc()

        # ── HMM ──────────────────────────────────────────────
        try:
            with open(hmm_path, "rb") as f:
                self.hmm = pickle.load(f)
            self.n_hmm_states = self.hmm.n_components
            self.hmm_ok = True
            logger.info(
                f"HMM loaded: {type(self.hmm).__name__} "
                f"| {self.n_hmm_states} states "
                f"| {self.hmm.n_features} obs"
            )
        except Exception as e:
            logger.warning(f"HMM load failed: {e} — using heuristic fallback")

    def predict_hmm(self, obs_sequence: list) -> tuple:
        """
        Run HMM on observation sequence.
        Returns (state_name, state_probs_list).
        """
        if not self.hmm_ok or not obs_sequence:
            # Fallback: map observation to state heuristically
            last_obs = obs_sequence[-1] if obs_sequence else 0
            state_idx = min(last_obs // 2, len(HMM_STATES) - 1)
            probs = [0.1] * len(HMM_STATES)
            probs[state_idx] = 0.6
            total = sum(probs)
            probs = [p / total for p in probs]
            return HMM_STATES[state_idx], probs

        try:
            obs_array = np.array(obs_sequence).reshape(-1, 1)
            # Predict most likely hidden state sequence
            states = self.hmm.predict(obs_array)
            state_probs_raw = self.hmm.predict_proba(obs_array)
            # Use last timestep's state and probs
            last_state_idx  = int(states[-1])
            last_state_probs = state_probs_raw[-1].tolist()

            # Map HMM state index to kill chain stage name
            # HMM states may not be labeled — use index mapping
            n = min(last_state_idx, len(HMM_STATES) - 1)
            state_name = HMM_STATES[n]

            # Pad or trim probs to match HMM_STATES length
            if len(last_state_probs) >= len(HMM_STATES):
                probs = last_state_probs[:len(HMM_STATES)]
            else:
                probs = last_state_probs + [0.0] * (len(HMM_STATES) - len(last_state_probs))

            return state_name, probs
        except Exception as e:
            logger.error(f"HMM predict error: {e}")
            return "Reconnaissance", [0.6, 0.2, 0.1, 0.05, 0.05]

    def predict_lstm(self, obs_sequence: list) -> tuple:
        """
        Run LSTM on observation sequence.
        Returns (state_name, confidence, state_probs, next_obs_name).
        """
        if not self.lstm_ok:
            return "Reconnaissance", 0.85, [0.6, 0.2, 0.1, 0.05, 0.05], "port_scan"

        try:
            # Pad sequence to match training config (seq_length - 1)
            max_len = self.seq_length
            padded = np.zeros(max_len, dtype=np.int64)
            hist = np.array(obs_sequence[-max_len:], dtype=np.int64)
            padded[max_len - len(hist):] = hist

            tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                obs_logits, state_logits = self.lstm(tensor)
                state_probs = torch.softmax(state_logits, dim=-1).squeeze().cpu().numpy()
                obs_probs = torch.softmax(obs_logits, dim=-1).squeeze().cpu().numpy()

            state_idx = int(state_probs.argmax())
            state_name = HMM_STATES[state_idx] if state_idx < len(HMM_STATES) else "Reconnaissance"
            confidence = float(state_probs.max())
            
            # Get predicted next observation
            obs_idx = int(obs_probs.argmax())
            obs_names = list(OBS_TO_ID.keys())
            next_obs = obs_names[obs_idx] if obs_idx < len(obs_names) else "port_scan"

            return state_name, confidence, state_probs.tolist(), next_obs
        except Exception as e:
            logger.error(f"LSTM predict error: {e}")
            import traceback
            traceback.print_exc()
            return "Reconnaissance", 0.75, [0.6, 0.2, 0.1, 0.05, 0.05], "port_scan"


models = L5Models()


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="IMMUNEX Layer 5 — Threat Memory", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class PredictRequest(BaseModel):
    chain_id:      Optional[str]    = None
    action:        Optional[str]    = "monitor"
    target_ip:     Optional[str]    = "unknown"
    attack_type:   Optional[str]    = None
    mitre_stage:   Optional[str]    = None
    verified_safe: Optional[bool]   = False
    q_value:       Optional[float]  = 0.5
    # Optional extras L3 might send
    nodes:         Optional[Any]    = None
    edges:         Optional[Any]    = None
    confidence:    Optional[float]  = 0.5


@app.on_event("startup")
async def startup():
    lstm_path = os.path.join(BASE_DIR, "immunex_lstm_final.pt")
    hmm_path  = os.path.join(BASE_DIR, "immunex_hmm.pkl")
    models.load(lstm_path, hmm_path)
    logger.info(
        f"Layer 5 ready | LSTM={'✅' if models.lstm_ok else '⚠️ fallback'} "
        f"| HMM={'✅' if models.hmm_ok else '⚠️ fallback'}"
    )


@app.get("/health")
def health():
    db_stats = threat_db.get_stats()
    return {
        "status": "ok",
        "layer":  5,
        "device": str(models.device),
        "lstm":   "loaded" if models.lstm_ok  else "fallback",
        "hmm":    "loaded" if models.hmm_ok   else "fallback",
        "persistence": "sqlite",
        "db_stats": db_stats
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        # ── Build observation sequence from available context ─
        attack_type = req.attack_type or ""
        mitre_stage = req.mitre_stage or ""
        target_ip = req.target_ip or "unknown"
        chain_id = req.chain_id or f"chain_{target_ip}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        obs_name = ATTACK_TO_OBS.get(attack_type, None)
        if not obs_name:
            # Infer from MITRE stage
            stage_obs_map = {
                "Reconnaissance": "port_scan",
                "Initial_Access":  "login_fail",
                "Initial Access":   "login_fail",
                "Execution":       "lateral_movement",
                "Impact":          "large_upload",
                "Exfiltration":    "large_upload",
                "Exploitation":    "priv_escalation",
            }
            obs_name = stage_obs_map.get(mitre_stage, "port_scan")

        obs_id  = OBS_TO_ID.get(obs_name, 0)

        # ── FIX 8: Persist observation and get historical sequence ─
        # Get or create the attack chain
        threat_db.get_or_create_chain(chain_id, target_ip)
        
        # Add this observation to the chain
        threat_db.add_observation(chain_id, obs_name, obs_id, attack_type)
        
        # Get historical observation sequence from DB (real sequence, not synthetic)
        obs_sequence = threat_db.get_observation_sequence(chain_id, limit=50)
        
        # If sequence is too short, pad with synthetic prefix
        if len(obs_sequence) < 3:
            OBS_SEQUENCES = {
                "port_scan":        [0, 0, 1],
                "dns_query":        [1, 0, 1],
                "phishing_click":   [0, 1, 2],
                "login_fail":       [0, 2, 3],
                "login_success":    [3, 3, 4],
                "priv_escalation":  [3, 4, 5],
                "lateral_movement": [4, 5, 6],
                "file_access":      [6, 5, 7],
                "large_upload":     [7, 6, 8],
                "zip_creation":     [7, 8, 9],
            }
            synthetic = OBS_SEQUENCES.get(obs_name, [obs_id])
            obs_sequence = synthetic[:-1] + obs_sequence  # Prepend synthetic context

        # ── HMM: infer kill chain stage ───────────────────────
        hmm_state, hmm_state_probs = models.predict_hmm(obs_sequence)

        # ── LSTM: attack progression prediction ───────────────
        lstm_state, lstm_conf, lstm_state_probs, next_obs = models.predict_lstm(obs_sequence)

        # Use LSTM state as primary (better accuracy from training)
        # HMM provides second opinion
        current_state = lstm_state
        
        # Blend confidences
        hmm_conf    = max(hmm_state_probs) if hmm_state_probs else 0.7
        confidence  = round((hmm_conf * 0.4 + lstm_conf * 0.6), 4)

        # FIX C: Disagreement handling - when LSTM and HMM disagree by 2+ stages, escalate
        STAGE_ORDER = ["Reconnaissance", "Initial_Access", "Privilege_Escalation", 
                       "Lateral_Movement", "Exfiltration"]
        agreement = (lstm_state == hmm_state)
        escalate_to_soc = False
        escalation_reason = None
        
        if not agreement:
            lstm_idx = STAGE_ORDER.index(lstm_state) if lstm_state in STAGE_ORDER else 0
            hmm_idx  = STAGE_ORDER.index(hmm_state)  if hmm_state  in STAGE_ORDER else 0
            stage_gap = abs(lstm_idx - hmm_idx)
            
            # Use the MORE ADVANCED stage (higher index = further in kill chain)
            current_state = STAGE_ORDER[max(lstm_idx, hmm_idx)]
            
            if stage_gap >= 2:
                risk = "HIGH"    # Large disagreement = genuine uncertainty, escalate
                escalate_to_soc = True
                escalation_reason = f"LSTM/HMM disagreement - stage gap of {stage_gap}"
            elif stage_gap == 1:
                risk = "MEDIUM"
                escalate_to_soc = False
            else:
                risk = STATE_RISK.get(current_state, "MEDIUM")
        else:
            risk = STATE_RISK.get(current_state, "MEDIUM")

        # ── Build response ────────────────────────────────────────
        threats  = STATE_TO_THREATS.get(current_state, ["Unknown threat vector"])
        playbook = STATE_TO_PLAYBOOK.get(current_state, "Monitor and investigate.")
        timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "chain_id":          chain_id,
            "current_state":     current_state,
            "predicted_threats": threats,
            "time_window":       "2-6 hours",
            "confidence":        confidence,
            "lstm_confidence":   round(lstm_conf, 4),
            "lstm_state":        lstm_state,
            "lstm_state_probs":  [round(p, 4) for p in lstm_state_probs],
            "hmm_confidence":    round(hmm_conf, 4),
            "hmm_state":         hmm_state,
            "hmm_state_probs":   [round(p, 4) for p in hmm_state_probs],
            "hmm_state_labels":  HMM_STATES[:len(hmm_state_probs)],
            "agreement":         agreement,
            "predicted_next_obs": next_obs,
            "playbook":          playbook,
            "risk_level":        risk,
            "observation":       obs_name,
            "attack_type":       attack_type,
            "target_ip":         target_ip,
            "timestamp":         timestamp,
            "observation_count": len(obs_sequence),
            "escalated_to_soc":  escalate_to_soc,
            "escalation_reason": escalation_reason,
        }

        # ── FIX 8: Persist prediction and update chain state ─
        threat_db.update_chain_state(chain_id, current_state, risk)
        threat_db.log_prediction(result)
        threat_db.update_attacker_profile(target_ip, chain_id, current_state)

        logger.info(
            f"Predicted {chain_id} | state={current_state} "
            f"| risk={risk} | conf={confidence:.3f} | obs_count={len(obs_sequence)}"
        )
        return result

    except Exception as e:
        logger.error(f"Predict failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Orchestrator calls /explain — alias to /predict
@app.post("/explain")
async def explain(req: PredictRequest):
    return await predict(req)


# ─────────────────────────────────────────────────────────────
# FIX 8: Additional endpoints for persistence features
# ─────────────────────────────────────────────────────────────

@app.get("/chains")
async def get_active_chains(limit: int = 100):
    """Get all active attack chains."""
    chains = threat_db.get_active_chains(limit)
    return {"chains": chains, "count": len(chains)}


@app.get("/chain/{chain_id}")
async def get_chain_history(chain_id: str):
    """Get full history for a specific chain."""
    history = threat_db.get_chain_history(chain_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Chain {chain_id} not found")
    return history


@app.get("/attacker/{ip}")
async def get_attacker_profile(ip: str):
    """Get attacker profile by IP."""
    profile = threat_db.get_attacker_profile(ip)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No profile for IP {ip}")
    return profile


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    return threat_db.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8005, reload=False)