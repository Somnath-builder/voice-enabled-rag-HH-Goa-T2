from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import query, voice, benchmark

app = FastAPI(
    title="HH Goa 2026 - Voice RAG",
    description="Backend for Voice-Enabled RAG System",
    version="1.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
