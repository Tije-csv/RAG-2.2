from typing import Dict, Any
import time
import logging
from dataclasses import dataclass
import prometheus_client as prom

@dataclass
class RAGMetrics:
    retrieval_time: float = 0
    rerank_time: float = 0
    generation_time: float = 0
    num_chunks_retrieved: int = 0

class RAGMonitor:
    def __init__(self):
        # Prometheus metrics
        self.retrieval_latency = prom.Histogram('rag_retrieval_latency_seconds', 'Retrieval time')
        self.rerank_latency = prom.Histogram('rag_rerank_latency_seconds', 'Reranking time')
        self.query_counter = prom.Counter('rag_queries_total', 'Total queries processed')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('RAGMonitor')

    def log_query(self, query: str, metrics: RAGMetrics):
        self.query_counter.inc()
        self.retrieval_latency.observe(metrics.retrieval_time)
        self.rerank_latency.observe(metrics.rerank_time)
        
        self.logger.info({
            'query': query,
            'metrics': metrics.__dict__
        })

    def evaluate_retrieval(self, retrieved_docs: List[Dict], relevant_docs: List[str]) -> Dict[str, float]:
        # Calculate precision and recall
        relevant_set = set(relevant_docs)
        retrieved_set = set(doc['text'] for doc in retrieved_docs)
        
        precision = len(relevant_set & retrieved_set) / len(retrieved_set)
        recall = len(relevant_set & retrieved_set) / len(relevant_set)
        
        return {'precision': precision, 'recall': recall}
