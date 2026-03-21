import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
import pickle
import logging
from typing import List, Tuple, Optional
from shared.redis_client import IMMUNEXCache

logger = logging.getLogger(__name__)

INDEX_PATH  = "models/faiss_index.bin"
META_PATH   = "models/faiss_meta.pkl"
DIMENSION   = 768   # RoBERTa CLS embedding dimension
NLIST       = 100   # IVF number of clusters
NPROBE      = 10    # clusters to search at query time
THRESHOLD   = 0.85  # similarity threshold for anomaly


class FAISSIndex:
    """
    IVF-PQ FAISS index for fast embedding similarity search.
    Stores RoBERTa embeddings of known-normal traffic.
    High distance from normal = anomaly.
    """

    def __init__(self, use_gpu=False):
        self.dimension  = DIMENSION
        self.index      = None
        self.metadata   = []   # list of dicts per embedding
        self.cache      = IMMUNEXCache()
        self.use_gpu    = use_gpu
        self._build_index()
        logger.info(f"FAISS IVF-PQ index initialized (GPU={use_gpu})")

    def _build_index(self):
        """Build IVF-PQ index. Load from disk if exists."""
        if os.path.exists(INDEX_PATH):
            self.load()
        else:
            # Flat index for when we have < 1000 vectors
            # Will upgrade to IVF-PQ after first bulk add
            self.index    = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            logger.info("New FAISS index created (flat, will upgrade to IVF-PQ)")

    def _upgrade_to_ivfpq(self, vectors: np.ndarray):
        """Upgrade flat index to IVF-PQ when we have enough vectors."""
        logger.info(f"Upgrading to IVF-PQ with {len(vectors)} vectors...")
        quantizer = faiss.IndexFlatL2(self.dimension)
        index_ivf = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            NLIST,   # number of clusters
            8,       # bytes per vector (compression)
            8        # bits per byte
        )
        index_ivf.nprobe = NPROBE
        index_ivf.train(vectors)
        index_ivf.add(vectors)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
                logger.info("FAISS running on GPU")
            except Exception as e:
                logger.warning(f"GPU failed, using CPU: {e}")
                self.index = index_ivf
        else:
            self.index = index_ivf

        logger.info("Upgraded to IVF-PQ index")

    def add_normal(self, embedding: List[float], metadata: dict):
        """Add a known-normal traffic embedding to the index."""
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        current_size = self.index.ntotal

        # Upgrade to IVF-PQ at 1000 vectors
        if current_size >= 1000 and isinstance(self.index, faiss.IndexFlatL2):
            all_vecs = faiss.rev_swig_ptr(self.index.xb.data(), current_size * self.dimension)
            all_vecs = all_vecs.reshape(current_size, self.dimension).copy()
            new_vecs = np.vstack([all_vecs, vec])
            self._upgrade_to_ivfpq(new_vecs)
            self.metadata.append(metadata)
        else:
            self.index.add(vec)
            self.metadata.append(metadata)

    def bulk_add_normal(self, embeddings: List[List[float]], metadatas: List[dict]):
        """Add many normal embeddings at once — use after initial training."""
        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)

        if len(embeddings) >= NLIST and isinstance(self.index, faiss.IndexFlatL2):
            self._upgrade_to_ivfpq(vecs)
        else:
            self.index.add(vecs)

        self.metadata.extend(metadatas)
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")

    def search(self, embedding: List[float], k=5) -> Tuple[List[float], List[dict]]:
        """
        Search for k nearest neighbors.
        Returns (distances, metadata_list).
        Lower distance = more similar to normal = less anomalous.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — skipping similarity search")
            return [999.0], [{}]

        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        distances = distances[0].tolist()
        metas = [self.metadata[i] if i < len(self.metadata) else {} 
                 for i in indices[0].tolist()]
        return distances, metas

    def is_anomalous(self, embedding: List[float]) -> Tuple[bool, float]:
        """
        Returns (is_anomalous, anomaly_score).
        High distance from normal neighbors = anomalous.
        """
        distances, _ = self.search(embedding, k=5)
        avg_distance  = float(np.mean(distances))
        # Normalize to 0-1 score (higher = more anomalous)
        anomaly_score = min(1.0, avg_distance / 2.0)
        is_anomalous  = anomaly_score > THRESHOLD
        return is_anomalous, anomaly_score

    def save(self):
        """Save index and metadata to disk."""
        os.makedirs("models", exist_ok=True)
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, INDEX_PATH)
        else:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"FAISS index saved ({self.index.ntotal} vectors)")

    def load(self):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS loaded on GPU")
            except Exception as e:
                logger.warning(f"GPU load failed, using CPU: {e}")
        logger.info(f"FAISS index loaded ({self.index.ntotal} vectors)")

    def build_from_training_data(self, roberta_model, tokenizer, 
                                  train_csv="master_dataset/roberta_train.csv",
                                  max_normal=50000):
        """
        Build index from training data.
        Runs RoBERTa on normal traffic to build the normal baseline.
        """
        import pandas as pd
        import torch

        logger.info("Building FAISS index from training data...")
        df = pd.read_csv(train_csv)
        normal = df[df["label"] == 0].head(max_normal)
        logger.info(f"Processing {len(normal)} normal samples...")

        device = next(roberta_model.parameters()).device
        embeddings = []
        batch_size = 64

        for i in range(0, len(normal), batch_size):
            batch_texts = normal["text"].iloc[i:i+batch_size].tolist()
            enc = tokenizer(
                batch_texts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                out = roberta_model.roberta(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device)
                )
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls.tolist())

            if i % 5000 == 0:
                logger.info(f"Processed {i}/{len(normal)} samples")

        metadatas = [{"label": "normal", "idx": i} for i in range(len(embeddings))]
        self.bulk_add_normal(embeddings, metadatas)
        self.save()
        logger.info(f"FAISS index built with {len(embeddings)} normal embeddings")