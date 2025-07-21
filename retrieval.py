from typing import List, Dict
import faiss
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vector_dim: int = 128):
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(vector_dim), vector_dim, 4)
        self.bm25 = None
        self.documents = []
        self.metadata = []

    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        self.documents = documents
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.metadata = metadata or [{} for _ in documents]

    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        # Dense search
        dense_scores, dense_ids = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # Sparse search
        sparse_scores = self.bm25.get_scores(query.split())
        sparse_ids = np.argsort(sparse_scores)[-top_k:]
        
        # Combine results
        combined_ids = list(set(dense_ids[0].tolist() + sparse_ids.tolist()))
        results = [
            {
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(sparse_scores[idx])
            }
            for idx in combined_ids[:top_k]
        ]
        return results
