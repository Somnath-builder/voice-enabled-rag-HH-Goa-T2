from fastembed import TextEmbedding
import numpy as np
from app.config import settings

class EmbeddingService:
    def __init__(self):
        print(f"Loading lightweight embedding model: {settings.EMBEDDING_MODEL}")
        # FastEmbed uses optimized ONNX runtime and has zero PyTorch dependency!
        self.model = TextEmbedding(settings.EMBEDDING_MODEL)
    
    def embed_query(self, query: str) -> np.ndarray:
        # FastEmbed returns a generator of arrays, we just take the first one
        embedding = list(self.model.embed([query]))[0]
        return embedding.astype("float32")

# Singleton instance
embedding_service = EmbeddingService()
