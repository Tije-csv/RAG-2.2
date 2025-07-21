import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def create_initial_corpus():
    # Sample documents (replace these with your actual documents)
    documents = [
        "This is a sample document for the RAG system.",
        "Another example document to populate the corpus.",
        "Information retrieval systems help find relevant content.",
        "FAISS is an efficient similarity search library.",
        "Vector embeddings represent text in high-dimensional space."
    ]

    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    # Create FAISS index
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 4)
    
    # Need at least 4 vectors to train
    if len(documents) < 4:
        print("Warning: Need at least 4 documents for IVF training")
        return
    
    # Train and add vectors to the index
    index.train(embeddings)
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, "corpus_index.faiss")

    # Create and save corpus data
    corpus_data = {
        "documents": documents,
        "metadata": [{"id": i, "source": "initial_corpus"} for i in range(len(documents))]
    }

    with open("corpus_data.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    create_initial_corpus()
    print("Created corpus_index.faiss and corpus_data.json with initial data")