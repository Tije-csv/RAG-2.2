from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pickle
import redis
from pathlib import Path
from media_loader import MediaLoader
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class EnhancedRAGPipeline:
    def __init__(self, api_key: str, redis_host: str = 'localhost', redis_port: int = 6379, redis_password: str = ''):
        self.query_classifier = pipeline("zero-shot-classification")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize Redis with configuration
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.setup_indexes()

    def setup_indexes(self):
        # Initialize FAISS index with reduced dimensions
        original_dim = 384
        self.target_dim = 128
        self.pca_matrix = faiss.PCAMatrix(original_dim, self.target_dim)
        self.index = faiss.IndexIVFFlat(self.pca_matrix, self.target_dim, 4)
        
    def needs_retrieval(self, query: str) -> bool:
        # Classify if query needs retrieval
        labels = ["factual", "generative"]
        result = self.query_classifier(query, labels)
        return result["labels"][0] == "factual"

    def process_query(self, query: str) -> Dict[str, Any]:
        if not self.needs_retrieval(query):
            # Use Gemini for direct generation
            response = self.model.generate_content(query)
            return {"response": response.text}
            
        # Get cached results if available
        cache_key = f"query:{hash(query)}"
        if cached := self.redis_client.get(cache_key):
            return pickle.loads(cached)

        # Perform retrieval and reranking
        retrieved_docs = self.retrieve_and_rerank(query)
        
        # Construct prompt with retrieved context
        context = "\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nAnswer based on the context provided:"""
        
        # Generate response using Gemini
        response = self.model.generate_content(prompt)
        
        results = {
            "response": response.text,
            "retrieved_docs": retrieved_docs
        }
        
        # Cache results
        self.redis_client.setex(cache_key, 3600, pickle.dumps(results))
        return results

    def add_media_documents(self, file_paths: List[str] = None, directory_path: str = None) -> None:
        """Add documents from media files to the RAG pipeline."""
        documents = []
        
        if directory_path:
            documents.extend(self.media_loader.load_directory(directory_path))
        
        if file_paths:
            for file_path in file_paths:
                content = self.media_loader.load_file(file_path)
                if content:
                    documents.append({
                        'content': content,
                        'source': file_path,
                        'type': Path(file_path).suffix[1:]
                    })
        
        if documents:
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedding_model.encode(texts)
            self.index.add(embeddings)
            # Store document metadata
            for doc, emb in zip(documents, embeddings):
                key = f"doc:{hash(doc['content'])}"
                self.redis_client.hset(key, mapping={
                    'content': doc['content'],
                    'source': doc['source'],
                    'type': doc['type'],
                    'embedding': emb.tobytes()
                })
