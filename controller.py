import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import DBSCAN


class E5Embedder:
    """Minimal wrapper for intfloat/multilingual-e5-large-instruct."""

    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def instruct(task: str, query: str) -> str:
        return f"Instruct: {task}\nQuery: {query}"

    @staticmethod
    def _avg_pool(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = h.masked_fill(~m[..., None].bool(), 0.0) # ~ is the bitwise not op
        return h.sum(1) / m.sum(1, keepdim=True)

    def encode(self, texts, max_len=512, batch_size=32):
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                t = self.tok(texts[i:i + batch_size], padding=True,
                             truncation=True, max_length=max_len,
                             return_tensors="pt").to(self.device)
                h = self.model(**t).last_hidden_state
                e = F.normalize(self._avg_pool(h, t["attention_mask"]), p=2, dim=1)
                embs.append(e.cpu())
        return torch.cat(embs)
        
import numpy as np

class VectorDB:
    """Tiny, NumPy-only cosine-similarity DB with E5 instructions."""
    def __init__(self, embedder):
        self.e = embedder
        self.task = "Given a concept, find the most similar concepts"
        self.vecs = np.empty((0, self.e.model.config.hidden_size), np.float32)
        self.ids, self.texts = [], []

    # ---------- helpers ----------
    @staticmethod
    def _l2norm(v):
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v.to(torch.float32)

    # ---------- indexing ----------
    def add_docs(self, texts, batch_size=64):
        vecs = self._l2norm(self.e.encode(texts, batch_size=batch_size))
        self.vecs = np.concatenate([self.vecs, vecs], axis=0)
        start = len(self.ids)
        self.ids.extend(range(start, start + len(texts)))
        self.texts.extend(texts)

    # ---------- search ----------
    def _qvec(self, q):
        q_vec = self.e.encode([self.e.instruct(self.task, q)])[0]
        return q_vec / (np.linalg.norm(q_vec) + 1e-9)

    def search(self, query, k=5):
        q = self._qvec(query)
        sims = q @ self.vecs.T
        
        # Ensure k doesn't exceed the size of the array
        k = min(k, len(self.vecs))
        
        # If there are no vectors, return empty list
        if k == 0:
            return []
        
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in idx]

class Controller:
    """Minimal wrapper for controller component of the network"""
    def __init__(self):
        self.vdb = VectorDB(embedder=E5Embedder())
        self.threshold = 0.8
    def add_concept(self, concept): self.vdb.add_docs(concept)

    def similar(self, concept): 
        """ Check if there exists similar concept beyond threshold """
        results = self.vdb.search(concept)
        if len(results) == 0:
            return False
        (_, _, similarity) = results[0]
        if similarity > self.threshold:
            return True
        return False

    def select(self, k, knn=10):
        V = self.vdb.vecs            # (N, d)
        if V.size == 0: return []

        # pairwise dot‑product → cosine sim
        sims = V @ V.T              # (N, N)
        np.fill_diagonal(sims, -1)  # ignore self

        # collect k‑NN distances (cosine → distance = 1‑sim)
        D = 1 - np.sort(sims, axis=1)[:, -knn:]   # largest sims = nearest
        density = 1 / (D.mean(1) + 1e-9)          # higher = denser

        top = np.argpartition(-density, k)[:k]
        top = top[np.argsort(-density[top])]

        return [self.vdb.texts[i] for i in top]