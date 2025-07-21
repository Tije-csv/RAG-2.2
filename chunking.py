import nltk
from typing import List, Text
import spacy

class HybridChunker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.overlap_size = 50
        self.chunk_size = 250

    def fixed_length_chunks(self, text: str) -> List[str]:
        tokens = text.split()
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap_size):
            chunk = " ".join(tokens[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def semantic_chunks(self, text: str) -> List[str]:
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_length = len(sent.text.split())
            if current_length + sent_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent.text]
                current_length = sent_length
            else:
                current_chunk.append(sent.text)
                current_length += sent_length
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
