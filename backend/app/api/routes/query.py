import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.orchestrator import orchestrator_service

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
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    result = await orchestrator_service.process_query(request.query)
    
    return QueryResponse(**result)
