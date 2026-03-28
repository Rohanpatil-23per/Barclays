import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from layer1_detection.inference_utils import load_top_features, serialize_features

class Layer1Detector:
    def __init__(self,
                 cicids_roberta_path="models/roberta_layer1",
                 unsw_roberta_path="models/roberta_unsw",
                 iso_path="models/isolation_forest.pkl",
                 cicids_scaler_path="models/layer1_scaler.pkl",
                 unsw_scaler_path="models/unsw_scaler.pkl",
                 stats_path="models/iso_forest_stats.json",
                 cicids_feats_path="master_dataset/top_features.json",
                 unsw_feats_path="models/unsw_features.json"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Layer1Detector on {self.device}...")

        # CICIDS RoBERTa
        self.cicids_tokenizer = RobertaTokenizer.from_pretrained(cicids_roberta_path)
        self.cicids_roberta   = RobertaForSequenceClassification.from_pretrained(cicids_roberta_path)
        self.cicids_roberta.to(self.device)
        self.cicids_roberta.eval()

        # UNSW RoBERTa (loads only if trained)
        self.unsw_available = os.path.exists(unsw_roberta_path)
        if self.unsw_available:
            self.unsw_tokenizer = RobertaTokenizer.from_pretrained(unsw_roberta_path)
            self.unsw_roberta   = RobertaForSequenceClassification.from_pretrained(unsw_roberta_path)
            self.unsw_roberta.to(self.device)
            self.unsw_roberta.eval()
            with open(unsw_scaler_path, "rb") as f:
                self.unsw_scaler = pickle.load(f)
            with open(unsw_feats_path) as f:
                self.unsw_features = json.load(f)
            print("UNSW model loaded ✅")
        else:
            print("UNSW model not found, using CICIDS only")

        # Isolation Forest
        with open(iso_path, "rb") as f:
            self.iso_forest = pickle.load(f)
        with open(cicids_scaler_path, "rb") as f:
            self.cicids_scaler = pickle.load(f)

        self.cicids_features = load_top_features(cicids_feats_path)

        with open(stats_path) as f:
            self.iso_threshold = json.load(f)["threshold"]

        print("Layer1Detector ready ✅")

    def _roberta_score(self, model, tokenizer, text):
        """Run one RoBERTa model, return attack_prob and embedding."""
        enc = tokenizer(text, max_length=128, padding='max_length',
                        truncation=True, return_tensors='pt')
        with torch.no_grad():
            out         = model(
                input_ids      = enc['input_ids'].to(self.device),
                attention_mask = enc['attention_mask'].to(self.device)
            )
            attack_prob = float(torch.softmax(out.logits, dim=1)[0][1].cpu())
            embedding   = model.roberta(
                input_ids      = enc['input_ids'].to(self.device),
                attention_mask = enc['attention_mask'].to(self.device)
            ).last_hidden_state[:, 0, :].squeeze().cpu().tolist()
        return attack_prob, embedding

    def detect(self, features: list) -> dict:
        """
        Input:  features — list of 77 floats (CICIDS format)
        Output: dict with anomaly_score, is_anomalous, embedding, method
        """
        arr    = np.array(features).reshape(1, -1)
        scaled = self.cicids_scaler.transform(arr)

        # Isolation Forest
        iso_score = float(self.iso_forest.decision_function(scaled)[0])
        iso_flag  = iso_score < self.iso_threshold

        # CICIDS RoBERTa
        cicids_text         = serialize_features(features, self.cicids_features)
        cicids_prob, embedding = self._roberta_score(
            self.cicids_roberta, self.cicids_tokenizer, cicids_text
        )

        # UNSW RoBERTa (ensemble if available)
        if self.unsw_available:
            # Use first 39 features mapped to UNSW space
            unsw_arr    = self.unsw_scaler.transform(arr[:, :39])
            unsw_text   = " ".join(
                f"{n.lower().replace(' ','_')}:{unsw_arr[0][i]:.4f}"
                for i, n in enumerate(self.unsw_features[:25])
            )
            unsw_prob, _ = self._roberta_score(
                self.unsw_roberta, self.unsw_tokenizer, unsw_text
            )
            # Ensemble: weighted average (CICIDS trained on more data)
            attack_prob = 0.6 * cicids_prob + 0.4 * unsw_prob
            method_base = "ensemble"
        else:
            attack_prob = cicids_prob
            method_base = "transformer"

        is_anomalous  = attack_prob > 0.5 or iso_flag
        anomaly_score = float(np.clip(
            max(attack_prob, float(iso_flag) * 0.8), 0.0, 1.0
        ))

        if iso_flag and attack_prob > 0.5:
            method = f"both_{method_base}"
        elif iso_flag:
            method = "isolation_forest"
        else:
            method = method_base

        return {
            "anomaly_score":    anomaly_score,
            "is_anomalous":     is_anomalous,
            "embedding":        embedding,
            "detection_method": method,
            "attack_prob":      attack_prob,
            "iso_score":        iso_score
        }


if __name__ == "__main__":
    import random
    detector = Layer1Detector()
    fake = [random.uniform(-1, 2) for _ in range(77)]
    result = detector.detect(fake)
    print(f"Anomaly score: {result['anomaly_score']:.4f}")
    print(f"Is anomalous:  {result['is_anomalous']}")
    print(f"Method:        {result['detection_method']}")
    print(f"Embedding dim: {len(result['embedding'])}")
