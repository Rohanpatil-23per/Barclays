import json
import numpy as np

def load_top_features(path="master_dataset/top_features.json"):
    with open(path) as f:
        names = json.load(f)
    # Convert to same format used during training: lowercase, spaces→underscores
    return [n.lower().replace(" ", "_").replace("/", "/") for n in names]

def serialize_features(features, top_feat_names):
    """Convert 77-dim feature vector to text string matching training format."""
    parts = []
    for i, name in enumerate(top_feat_names[:25]):
        if i < len(features):
            val = features[i]
            parts.append(f"{name}:{val:.4f}")
    return " ".join(parts)
