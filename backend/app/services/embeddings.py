from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings

class EmbeddingService:
    def __init__(self):
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def embed_query(self, query: str) -> np.ndarray:
        # Encode and normalize
        embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(embedding).astype("float32")

# Singleton instance
embedding_service = EmbeddingService()
