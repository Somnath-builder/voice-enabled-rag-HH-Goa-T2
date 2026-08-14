import time
from typing import Dict, Any
from app.config import settings
from app.services.llm import llm_service
from app.services.retriever import retriever_service

class OrchestratorService:
    async def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # 0. Translate Query (Multilingual Support for single-language embeddings)
        translated_query = await llm_service.translate_to_english(query)
        
        # 1. Validate Input (Guardrail)
        is_valid = await llm_service.validate_query(translated_query)
        if not is_valid:
            return {
                "answer": "I'm sorry, I can only answer questions related to the provided knowledge base.",
                "grounded": False,
                "sources": [],
                "retrieval_latency_ms": 0.0,
                "generation_latency_ms": 0.0,
                "total_latency_ms": (time.time() - start_time) * 1000
            }
            
        # 2. Retrieve
        retrieval_res = retriever_service.retrieve(translated_query)
        chunks = retrieval_res["chunks"]
        retrieval_latency = retrieval_res["latency_ms"]
        
        # 3. Retrieval Threshold Validation (Guardrail)
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
            
        # 4. Generate Answer
        gen_res = await llm_service.generate_answer(translated_query, query, valid_chunks)
        answer = gen_res["answer"]
        generation_latency = gen_res["latency_ms"]
        
        # 5. Hallucination Check (Guardrail)
        is_grounded = await llm_service.check_hallucination(answer, valid_chunks)
        
        if not is_grounded:
            answer = "I found some information, but I couldn't confidently formulate an answer strictly based on the retrieved context."
        
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
