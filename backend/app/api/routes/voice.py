from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from app.services.voice import voice_service
from app.services.orchestrator import orchestrator_service

router = APIRouter()

class VoiceResponse(BaseModel):
    transcript: str
    answer: str
    grounded: bool
    sources: List[Dict[str, Any]]
    stt_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

@router.post("/voice", response_model=VoiceResponse)
async def process_voice(
    file: UploadFile = File(...),
    language_code: str = Form("en-IN")
):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an audio file.")
        
    # 1. Speech-to-Text
    stt_res = await voice_service.transcribe(file, language_code)
    transcript = stt_res["transcript"]
    stt_latency = stt_res["latency_ms"]
    
    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio. Please try speaking clearly again.")
        
    # 2. Run RAG Pipeline via Orchestrator
    rag_res = await orchestrator_service.process_query(transcript)
    
    # 3. Combine response metrics
    total_latency = stt_latency + rag_res["total_latency_ms"]
    
    return VoiceResponse(
        transcript=transcript,
        answer=rag_res["answer"],
        grounded=rag_res["grounded"],
        sources=rag_res["sources"],
        stt_latency_ms=stt_latency,
        retrieval_latency_ms=rag_res["retrieval_latency_ms"],
        generation_latency_ms=rag_res["generation_latency_ms"],
        total_latency_ms=total_latency
    )
