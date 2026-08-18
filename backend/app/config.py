import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "groq/compound-mini") # Groq standard fast model
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "vector.index")
    CHUNK_MAPPING_PATH: str = os.path.join(DATA_DIR, "chunk_mapping.json")
    TOP_K: int = 3
    RETRIEVAL_THRESHOLD: float = 0.60

settings = Settings()
