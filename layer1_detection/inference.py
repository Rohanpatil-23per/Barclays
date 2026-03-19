import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from layer1_detection.inference_utils import load_top_features, serialize_features

class Layer1Detector:
    def __init__(self,
                 roberta_path="models/roberta_layer1",
                 iso_path="models/isolation_forest.pkl",
                 scaler_path="models/layer1_scaler.pkl",
                 stats_path="models/iso_forest_stats.json",
                 top_feats_path="master_dataset/top_features.json"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RoBERTa
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.roberta   = RobertaForSequenceClassification.from_pretrained(roberta_path)
        self.roberta.to(self.device)
        self.roberta.eval()

        # Isolation Forest
        with open(iso_path, "rb") as f:
            self.iso_forest = pickle.load(f)

        # Scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Features + threshold
        self.top_features = load_top_features(top_feats_path)
        with open(stats_path) as f:
            self.iso_threshold = json.load(f)["threshold"]

        print(f"Layer1Detector loaded on {self.device} ✅")

    def detect(self, features: list) -> dict:
        """
        Input:  features — list of 77 floats
        Output: dict with anomaly_score, is_anomalous, embedding, method
        """
        arr    = np.array(features).reshape(1, -1)
        scaled = self.scaler.transform(arr)

        # Isolation Forest
        iso_score = float(self.iso_forest.decision_function(scaled)[0])
        iso_flag  = iso_score < self.iso_threshold

        # RoBERTa
        text = serialize_features(features, self.top_features)
        enc  = self.tokenizer(text, max_length=128, padding='max_length',
                              truncation=True, return_tensors='pt')

        with torch.no_grad():
            out         = self.roberta(
                input_ids      = enc['input_ids'].to(self.device),
                attention_mask = enc['attention_mask'].to(self.device)
            )
            attack_prob = float(torch.softmax(out.logits, dim=1)[0][1].cpu())
            embedding   = self.roberta.roberta(
                input_ids      = enc['input_ids'].to(self.device),
                attention_mask = enc['attention_mask'].to(self.device)
            ).last_hidden_state[:, 0, :].squeeze().cpu().tolist()

        is_anomalous  = attack_prob > 0.5 or iso_flag
        anomaly_score = float(np.clip(max(attack_prob, float(iso_flag) * 0.8), 0.0, 1.0))

        if iso_flag and attack_prob > 0.5:
            method = "both"
        elif iso_flag:
            method = "isolation_forest"
        else:
            method = "transformer"

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
