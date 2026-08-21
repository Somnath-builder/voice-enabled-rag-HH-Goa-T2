import time
from typing import Dict, Any
from app.config import settings
from app.services.llm import llm_service
from app.services.retriever import retriever_service

class OrchestratorService:
    async def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
            
        # 1. Retrieve (FastEmbed directly supports multilingual queries, so we skip explicit LLM translation)
        retrieval_res = retriever_service.retrieve(query)
        chunks = retrieval_res["chunks"]
        retrieval_latency = retrieval_res["latency_ms"]
        
        # Filter by threshold
        valid_chunks = [c for c in chunks if c.get("score", 0) >= settings.RETRIEVAL_THRESHOLD]
        
        if not valid_chunks:
            return {
                "answer": "I couldn't find sufficient information in the knowledge base to answer that.",
                "grounded": False,
                "sources": [],
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": 0.0,
                "total_latency_ms": (time.time() - start_time) * 1000
            }
            
        # 2. Orchestrated Generation (JSON Structured Output with Harness)
        # This single call handles Guardrails (Safety), Hallucination (Relevance), and Answer Generation
        gen_res = await llm_service.orchestrated_generation(query, valid_chunks)
        generation_latency = gen_res.get("latency_ms", 0.0)
        
        is_safe = gen_res.get("is_safe", True)
        is_relevant = gen_res.get("is_relevant", True)
        answer = gen_res.get("answer", "")
        confidence = gen_res.get("confidence_score", 0.0)
        
        if not is_safe:
            answer = "I'm sorry, I cannot fulfill this request as it violates safety policies."
            is_grounded = False
        elif not is_relevant or confidence < 0.5:
            answer = "I found some information, but I couldn't confidently formulate an answer strictly based on the retrieved context."
            is_grounded = False
        else:
            is_grounded = True
            
        total_latency = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "grounded": is_grounded,
            "sources": [{"chunk_id": c.get("chunk_id"), "score": c.get("score"), "text": c.get("text")} for c in valid_chunks],
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": generation_latency,
            "total_latency_ms": total_latency
        }

orchestrator_service = OrchestratorService()
