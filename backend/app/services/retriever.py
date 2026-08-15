import faiss
import json
import time
from typing import List, Dict, Any
from app.config import settings
from app.services.embeddings import embedding_service

class RetrieverService:
    def __init__(self):
        print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        try:
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Warning: Could not load FAISS index. Make sure to build it first. Error: {e}")
            self.index = None
            
        print(f"Loading chunk mapping from {settings.CHUNK_MAPPING_PATH}...")
        try:
            with open(settings.CHUNK_MAPPING_PATH, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load mapping. Error: {e}")
            self.mapping = {}

    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> Dict[str, Any]:
        if not self.index or not self.mapping:
            return {"chunks": [], "latency_ms": 0}
            
        start_time = time.time()
        query_vector = embedding_service.embed_query(query)
        
        # FAISS expects a 2D array, so we reshape the 1D array to (1, D)
        query_vector = query_vector.reshape(1, -1)
        
        # FAISS search
        scores, indices = self.index.search(query_vector, top_k)
        
        chunks = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and str(idx) in self.mapping:
                chunk = self.mapping[str(idx)]
                chunk["score"] = float(scores[0][i])
                chunks.append(chunk)
                
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "chunks": chunks,
            "latency_ms": latency_ms
        }

retriever_service = RetrieverService()
