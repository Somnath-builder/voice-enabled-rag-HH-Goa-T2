import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.retriever import retriever_service
from app.services.llm import llm_service

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[Dict[str, Any]]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # 1. Retrieve
    retrieval_res = retriever_service.retrieve(request.query)
    chunks = retrieval_res["chunks"]
    retrieval_latency = retrieval_res["latency_ms"]
    
    # Check if we got enough relevant chunks
    # (Simplified guardrail for now: if no chunks or max score is very low)
    if not chunks:
        return QueryResponse(
            answer="I couldn't find sufficient information in the provided knowledge base to answer that.",
            grounded=False,
            sources=[],
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=0.0,
            total_latency_ms=(time.time() - start_time) * 1000
        )
        
    # 2. Generate
    gen_res = await llm_service.generate_answer(request.query, chunks)
    answer = gen_res["answer"]
    generation_latency = gen_res["latency_ms"]
    
    total_latency = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=answer,
        grounded=True,
        sources=[{"chunk_id": c.get("chunk_id"), "score": c.get("score"), "text": c.get("text")} for c in chunks],
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        total_latency_ms=total_latency
    )
