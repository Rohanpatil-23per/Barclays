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
from datetime import datetime, timezone
from typing import List, Optional, Any

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
# LSTM model — introspect architecture from weights at startup
# ─────────────────────────────────────────────────────────────
class ThreatLSTM(nn.Module):
    """
    Flexible LSTM that adapts to loaded weight shapes.
    Detects input_size, hidden_size, num_layers from state_dict.
    Output: probability distribution over HMM states (5 classes).
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def infer_lstm_architecture(state_dict: dict):
    """Infer LSTM dimensions from weight tensor shapes."""
    # weight_ih_l0 shape: (4*hidden, input)
    wih = state_dict.get("lstm.weight_ih_l0")
    if wih is None:
        # Try without 'lstm.' prefix
        wih = state_dict.get("weight_ih_l0")
    if wih is None:
        return None

    hidden_size = wih.shape[0] // 4
    input_size  = wih.shape[1]

    # Count layers by checking weight_ih_l0, l1, l2...
    num_layers = 0
    for k in state_dict:
        if "weight_ih_l" in k:
            num_layers += 1

    # FC output size
    fc_w = state_dict.get("fc.weight", state_dict.get("classifier.weight"))
    num_classes = fc_w.shape[0] if fc_w is not None else 5

    return {
        "input_size":  input_size,
        "hidden_size": hidden_size,
        "num_layers":  max(num_layers, 1),
        "num_classes": num_classes,
    }


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

    def load(self, lstm_path: str, hmm_path: str):
        # ── LSTM ─────────────────────────────────────────────
        try:
            raw = torch.load(lstm_path, map_location="cpu", weights_only=False)
            state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw.state_dict()

            arch = infer_lstm_architecture(state_dict)
            if arch:
                logger.info(f"LSTM arch: {arch}")
                self.lstm = ThreatLSTM(**arch).to(self.device)
                # Try strict load first, then non-strict
                try:
                    self.lstm.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    self.lstm.load_state_dict(state_dict, strict=False)
                    logger.warning("LSTM loaded with strict=False (partial weights)")
                self.lstm.eval()
                self.lstm_ok = True
                logger.info(f"LSTM loaded on {self.device}")
            else:
                logger.warning("Could not infer LSTM architecture — using HMM only")
        except Exception as e:
            logger.warning(f"LSTM load failed: {e} — HMM-only mode")

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

    def predict_lstm(self, obs_sequence: list) -> float:
        """
        Run LSTM on observation sequence.
        Returns attack progression confidence [0,1].
        """
        if not self.lstm_ok:
            return 0.85

        try:
            # One-hot encode observations
            n_obs = 10
            seq = np.zeros((len(obs_sequence), n_obs), dtype=np.float32)
            for i, obs in enumerate(obs_sequence):
                if 0 <= obs < n_obs:
                    seq[i, obs] = 1.0

            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.lstm(tensor)
                probs  = torch.softmax(logits, dim=-1).squeeze()
            # Confidence = max class prob (higher stage = more danger)
            return float(probs.max().cpu())
        except Exception as e:
            logger.error(f"LSTM predict error: {e}")
            return 0.75


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
    return {
        "status": "ok",
        "layer":  5,
        "device": str(models.device),
        "lstm":   "loaded" if models.lstm_ok  else "fallback",
        "hmm":    "loaded" if models.hmm_ok   else "fallback",
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        # ── Build observation sequence from available context ─
        attack_type = req.attack_type or ""
        mitre_stage = req.mitre_stage or ""

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
        # Build a short realistic sequence leading to this observation
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
        obs_sequence = OBS_SEQUENCES.get(obs_name, [obs_id])

        # ── HMM: infer kill chain stage ───────────────────────
        current_state, state_probs = models.predict_hmm(obs_sequence)

        # ── LSTM: attack progression confidence ───────────────
        lstm_confidence = models.predict_lstm(obs_sequence)

        # Blend confidences
        hmm_conf    = max(state_probs) if state_probs else 0.7
        confidence  = round((hmm_conf * 0.6 + lstm_confidence * 0.4), 4)

        # ── Build response ────────────────────────────────────
        threats  = STATE_TO_THREATS.get(current_state, ["Unknown threat vector"])
        playbook = STATE_TO_PLAYBOOK.get(current_state, "Monitor and investigate.")
        risk     = STATE_RISK.get(current_state, "MEDIUM")

        result = {
            "chain_id":          req.chain_id,
            "current_state":     current_state,
            "predicted_threats": threats,
            "time_window":       "2-6 hours",
            "confidence":        confidence,
            "lstm_confidence":   round(lstm_confidence, 4),
            "hmm_confidence":    round(hmm_conf, 4),
            "playbook":          playbook,
            "hmm_state_probs":   [round(p, 4) for p in state_probs],
            "hmm_state_labels":  HMM_STATES[:len(state_probs)],
            "risk_level":        risk,
            "observation":       obs_name,
            "attack_type":       attack_type,
            "target_ip":         req.target_ip,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Predicted {req.chain_id} | state={current_state} "
            f"| risk={risk} | conf={confidence:.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Predict failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Orchestrator calls /explain — alias to /predict
@app.post("/explain")
async def explain(req: PredictRequest):
    return await predict(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8005, reload=False)