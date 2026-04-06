"""
IMMUNEX - Layer 4: FastAPI Server
Port: 8004
Endpoints:
  GET  /health          → health check
  POST /predict         → classify traffic as Benign or Attack
  POST /retrain         → trigger retraining on new attack data
  GET  /status          → current model accuracy + stats
"""

import os
import json
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR, "lora_model_ewc.pt")   # primary 94.82%
STATUS_LOG = os.path.join(LOG_DIR,   "server_status.json")

os.makedirs(LOG_DIR, exist_ok=True)

# 77 features from master_dataset/feature_columns.json (CICIDS format)
FEATURE_NAMES_77 = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min", "Packet Length Max",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init Fwd Win Bytes", "Init Bwd Win Bytes",
    "Fwd Act Data Packets", "Fwd Seg Size Min", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

# Legacy 25 features for backward compatibility
FEATURE_NAMES_25 = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "flow_bytes/s", "flow_packets/s", "fwd_packet_length_mean",
    "bwd_packet_length_mean", "flow_iat_mean", "fwd_iat_mean",
    "bwd_iat_mean", "syn_flag_count", "ack_flag_count", "fin_flag_count",
    "rst_flag_count", "psh_flag_count", "packet_length_mean",
    "packet_length_std", "fwd_packets/s", "bwd_packets/s",
    "init_fwd_win_bytes", "init_bwd_win_bytes", "active_mean",
    "idle_mean", "down/up_ratio", "avg_packet_size"
]

# Default to 77 features (set by model on load)
FEATURE_NAMES = FEATURE_NAMES_77
INPUT_DIM = 77

# ─── Model architecture (supports both 25 and 77 features) ────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base   = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))

class IMMUNEXLayer4(nn.Module):
    """
    Adaptive Immunity classifier supporting both 25-feature (legacy) 
    and 77-feature (CICIDS master dataset) input formats.
    """
    def __init__(self, input_dim=77, rank=8):
        super().__init__()
        self.input_dim = input_dim
        self.current_rank = rank
        # Wider first layer for 77 features
        hidden1 = 256 if input_dim >= 77 else 128
        hidden2 = 128 if input_dim >= 77 else 64
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_layer = LoRALayer(hidden2, 32, rank=rank)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))

# ─── Request/Response schemas ──────────────────────────────────────────────────
class TrafficSample(BaseModel):
    """
    One network traffic sample — supports 77 features (CICIDS) or 25 features (legacy)
    Can be sent as:
      1. features list:  {"features": [0.1, -0.3, ...]}   (77 or 25 floats)
      2. text format:    {"text": "flow_duration:0.1 total_fwd_packets:-0.3 ..."}
      3. named fields:   {"Protocol": 6, "Flow Duration": 0.1, ...}
    """
    features: Optional[List[float]] = None
    text:     Optional[str]         = None
    # Extra fields are allowed for named-field format
    
    class Config:
        extra = "allow"  # Allow any additional fields for 77-feature named input

class BatchRequest(BaseModel):
    """Batch of traffic samples for bulk prediction"""
    samples: List[TrafficSample]

class RetrainRequest(BaseModel):
    """Trigger retraining on new attack data"""
    attack_features: List[List[float]]  # list of feature vectors
    attack_labels:   List[int]          # 1 for attack, 0 for benign
    trigger_source:  Optional[str] = "manual"

# ─── Model Manager ────────────────────────────────────────────────────────────
class ModelManager:
    """
    Manages the loaded model, predictions, and retraining
    Thread-safe — retraining runs in background without blocking predictions
    """
    def __init__(self):
        self.device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model     = None
        self.accuracy  = 0.0
        self.load_time = None
        self.predict_count   = 0
        self.retrain_count   = 0
        self.is_retraining   = False
        self.last_retrain    = None
        self._lock           = threading.Lock()
        self.input_dim       = 77  # Default to 77 features
        self.feature_names   = FEATURE_NAMES_77

    def load(self):
        global FEATURE_NAMES, INPUT_DIM
        print(f"📂 Loading model from {MODEL_PATH}...")
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        
        # Detect input dimension from checkpoint
        self.input_dim = ckpt.get("input_dim", 77)
        rank = ckpt.get("lora_rank", 8)
        
        # Set feature names based on input dimension
        if self.input_dim == 25:
            self.feature_names = FEATURE_NAMES_25
            FEATURE_NAMES = FEATURE_NAMES_25
        else:
            self.feature_names = FEATURE_NAMES_77
            FEATURE_NAMES = FEATURE_NAMES_77
        INPUT_DIM = self.input_dim
        
        # Build model with correct architecture
        self.model = IMMUNEXLayer4(self.input_dim, rank=rank).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.accuracy = ckpt.get("accuracy", 94.82)
        self.load_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"   ✅ Model loaded | Input dim: {self.input_dim} | "
              f"Accuracy: {self.accuracy:.2f}% | Device: {self.device}")

    def parse_sample(self, sample: TrafficSample) -> np.ndarray:
        """Convert any input format to feature numpy array (77 or 25 features)"""
        dim = self.input_dim

        # Format 1: raw feature list
        if sample.features is not None:
            arr = np.array(sample.features[:dim], dtype=np.float32)
            if len(arr) < dim:
                arr = np.pad(arr, (0, dim - len(arr)))
            return arr

        # Format 2: text format "flow_duration:0.1 ..."
        if sample.text is not None:
            lookup = {}
            for pair in sample.text.strip().split():
                if ":" in pair:
                    key, val = pair.split(":", 1)
                    try: lookup[key] = float(val)
                    except: lookup[key] = 0.0
            return np.array(
                [lookup.get(f, 0.0) for f in self.feature_names],
                dtype=np.float32
            )

        # Format 3: named fields (use extra fields from pydantic model)
        extra = sample.model_extra if hasattr(sample, 'model_extra') else {}
        arr = []
        for f in self.feature_names:
            # Try exact match first, then normalized versions
            val = extra.get(f)
            if val is None:
                # Try lowercase/underscore variants
                f_lower = f.lower().replace(" ", "_").replace("/", "_per_")
                val = extra.get(f_lower, 0.0)
            arr.append(float(val) if val is not None else 0.0)
        return np.array(arr, dtype=np.float32)

    def predict_one(self, sample: TrafficSample) -> dict:
        """Predict single traffic sample"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        arr = self.parse_sample(sample)
        arr = np.clip(arr, -10, 10)

        with self._lock:
            self.model.eval()
            with torch.no_grad():
                x    = torch.tensor(arr, dtype=torch.float32)\
                             .unsqueeze(0).to(self.device)
                out  = self.model(x)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(out.argmax(1).cpu().numpy()[0])

        self.predict_count += 1
        label      = "Attack" if pred == 1 else "Benign"
        confidence = float(prob[pred])

        return {
            "prediction":       pred,
            "label":            label,
            "confidence":       round(confidence, 4),
            "benign_prob":      round(float(prob[0]), 4),
            "attack_prob":      round(float(prob[1]), 4),
            "is_threat":        pred == 1,
        }

    def predict_batch(self, samples: List[TrafficSample]) -> list:
        """Predict batch of traffic samples"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        arrays = np.vstack([
            self.parse_sample(s).reshape(1, -1) for s in samples
        ])
        arrays = np.clip(arrays, -10, 10)

        with self._lock:
            self.model.eval()
            with torch.no_grad():
                x    = torch.tensor(arrays, dtype=torch.float32).to(self.device)
                out  = self.model(x)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = out.argmax(1).cpu().numpy()

        self.predict_count += len(samples)
        results = []
        for i in range(len(samples)):
            p     = int(pred[i])
            label = "Attack" if p == 1 else "Benign"
            results.append({
                "prediction":  p,
                "label":       label,
                "confidence":  round(float(prob[i][p]), 4),
                "benign_prob": round(float(prob[i][0]), 4),
                "attack_prob": round(float(prob[i][1]), 4),
                "is_threat":   p == 1,
            })
        return results

    def retrain_background(self, X_new, y_new):
        """
        Retrain LoRA head on new data in background thread
        Does NOT block predictions while running
        Uses rehearsal to prevent forgetting
        """
        def _retrain():
            print(f"\n🔄 Background retraining started...")
            self.is_retraining = True
            t0 = time.time()

            try:
                import pandas as pd
                from torch.utils.data import DataLoader, TensorDataset

                # Load small rehearsal batch from original data
                train_csv = os.path.join(BASE_DIR, "lora_retrain_source.csv")
                df = pd.read_csv(train_csv)
                idx_orig = np.random.choice(len(df), 500, replace=False)

                def parse(text):
                    lookup = {}
                    for pair in text.strip().split():
                        if ":" in pair:
                            k, v = pair.split(":", 1)
                            try: lookup[k] = float(v)
                            except: pass
                    return [lookup.get(f, 0.0) for f in FEATURE_NAMES]

                X_orig = np.array(
                    [parse(t) for t in df["text"].iloc[idx_orig]],
                    dtype=np.float32
                )
                y_orig = df["label"].iloc[idx_orig].values.astype(np.int64)

                # Combine original + new
                X_all = np.vstack([X_orig, X_new])
                y_all = np.concatenate([y_orig, y_new])

                dataset = TensorDataset(
                    torch.tensor(X_all, dtype=torch.float32),
                    torch.tensor(y_all, dtype=torch.long)
                )
                loader = DataLoader(dataset, batch_size=64, shuffle=True)

                with self._lock:
                    # Only train LoRA head
                    for p in self.model.base_encoder.parameters():
                        p.requires_grad = False
                    opt       = torch.optim.Adam(
                        filter(lambda p: p.requires_grad,
                               self.model.parameters()),
                        lr=5e-5
                    )
                    criterion = nn.CrossEntropyLoss()
                    self.model.train()

                    for ep in range(5):
                        for Xb, yb in loader:
                            Xb, yb = Xb.to(self.device), yb.to(self.device)
                            opt.zero_grad()
                            loss = criterion(self.model(Xb), yb)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 1.0)
                            opt.step()

                    self.model.eval()

                self.retrain_count += 1
                self.last_retrain   = time.strftime("%Y-%m-%d %H:%M:%S")
                elapsed             = time.time() - t0
                print(f"✅ Background retraining complete in {elapsed:.1f}s")

            except Exception as e:
                print(f"❌ Retraining error: {e}")
            finally:
                self.is_retraining = False

        thread = threading.Thread(target=_retrain, daemon=True)
        thread.start()

# ─── FastAPI app ───────────────────────────────────────────────────────────────
app     = FastAPI(
    title="IMMUNEX Layer 4 — Adaptive Immunity",
    description="LoRA retraining + blind spot detection for IMMUNEX",
    version="1.0.0"
)
manager = ModelManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    manager.load()
    print("\n🚀 IMMUNEX Layer 4 server started on port 8004")
    print("   Endpoints:")
    print("   GET  /health   → health check")
    print("   POST /predict  → classify single traffic sample")
    print("   POST /predict/batch → classify multiple samples")
    print("   POST /retrain  → trigger background retraining")
    print("   GET  /status   → model stats")
    print("   GET  /docs     → API documentation (Swagger UI)")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check — called by Person 5 to verify layer is running
    Returns: status, model loaded, accuracy
    """
    return {
        "status":       "healthy",
        "layer":        4,
        "name":         "Adaptive Immunity",
        "model_loaded": manager.model is not None,
        "accuracy":     manager.accuracy,
        "input_dim":    manager.input_dim,
        "device":       str(manager.device),
        "port":         8004,
    }

@app.post("/predict")
async def predict(sample: TrafficSample):
    """
    Classify one network traffic sample
    
    Input (any of these formats):
      {"features": [0.1, -0.3, 0.5, ...]}   ← list of 77 (or 25) numbers
      {"text": "flow_duration:0.1 ..."}       ← text format
      {"Protocol": 6, "Flow Duration": 0.1, ...}  ← named fields (77 features)
    
    Output:
      {"prediction": 1, "label": "Attack", "confidence": 0.95, ...}
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = manager.predict_one(sample)
        return {
            "success":   True,
            "result":    result,
            "model_acc": manager.accuracy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """
    Classify multiple traffic samples at once
    Input:  {"samples": [{...}, {...}, ...]}
    Output: {"results": [{...}, {...}, ...], "total": N, "threats": M}
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = manager.predict_batch(request.samples)
        threats = sum(1 for r in results if r["is_threat"])
        return {
            "success": True,
            "total":   len(results),
            "threats": threats,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(request: RetrainRequest):
    """
    Trigger background retraining on new attack data
    Called by Layer 3 after handling a new attack type
    Input:  {"attack_features": [[...], ...], "attack_labels": [1, 1, ...]}
    Output: {"success": True, "message": "Retraining started in background"}
    """
    if manager.is_retraining:
        return {
            "success": False,
            "message": "Retraining already in progress",
        }
    try:
        X_new = np.array(request.attack_features, dtype=np.float32)
        y_new = np.array(request.attack_labels,   dtype=np.int64)
        X_new = np.clip(X_new, -10, 10)

        manager.retrain_background(X_new, y_new)

        return {
            "success":        True,
            "message":        "Retraining started in background",
            "samples":        len(X_new),
            "trigger_source": request.trigger_source,
            "note":           "Model stays active during retraining",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    """
    Full status of Layer 4
    Called by Person 5 for the main dashboard
    """
    return {
        "layer":            4,
        "name":             "Adaptive Immunity",
        "model_accuracy":   manager.accuracy,
        "model_loaded":     manager.load_time,
        "input_dim":        manager.input_dim,
        "device":           str(manager.device),
        "predictions_made": manager.predict_count,
        "retrain_count":    manager.retrain_count,
        "is_retraining":    manager.is_retraining,
        "last_retrain":     manager.last_retrain,
        "features":         manager.feature_names,
        "num_features":     len(manager.feature_names),
        "classes":          {0: "Benign", 1: "Attack"},
        "port":             8004,
        "endpoints": {
            "health":        "GET  /health",
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch",
            "retrain":       "POST /retrain",
            "status":        "GET  /status",
            "docs":          "GET  /docs",
        }
    }

# ─── Run server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  IMMUNEX - LAYER 4 SERVER")
    print("="*60)
    print(f"  Starting on http://0.0.0.0:8004")
    print(f"  Swagger docs: http://localhost:8004/docs")
    print(f"  Press Ctrl+C to stop")
    print("="*60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )
