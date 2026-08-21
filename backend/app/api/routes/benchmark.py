from fastapi import APIRouter
import asyncio
import statistics
from typing import Dict, Any

from app.services.orchestrator import orchestrator_service
from app.services.retriever import retriever_service

router = APIRouter()

QUERIES = [
    "What is a corporation?",
    "How does tax law work?",
    "What are the rights of a shareholder?",
    "Explain intellectual property.",
    "What constitutes a breach of contract?",
]

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if pct == 100:
        return values[-1]
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])

@router.get("/benchmark")
async def run_benchmark() -> Dict[str, Any]:
    n = 5 # Reduced to 5 to avoid long timeout on HTTP request
    
    # Warmup
    retriever_service.retrieve("warmup query")
    
    total_ms = []
    retrieval_ms = []
    generation_ms = []
    
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = await orchestrator_service.process_query(query)
        
        total_ms.append(resp["total_latency_ms"])
        retrieval_ms.append(resp["retrieval_latency_ms"])
        generation_ms.append(resp["generation_latency_ms"])
        
        if i < n - 1:
            await asyncio.sleep(2.1) # Avoid Groq rate limits
            
    return {
        "queries_run": n,
        "metrics_ms": {
            "retrieval": {
                "avg": statistics.mean(retrieval_ms),
                "p50": percentile(retrieval_ms, 50),
                "p70": percentile(retrieval_ms, 70),
                "p100": percentile(retrieval_ms, 100),
            },
            "generation": {
                "avg": statistics.mean(generation_ms),
                "p50": percentile(generation_ms, 50),
                "p70": percentile(generation_ms, 70),
                "p100": percentile(generation_ms, 100),
            },
            "total_pipeline": {
                "avg": statistics.mean(total_ms),
                "p50": percentile(total_ms, 50),
                "p70": percentile(total_ms, 70),
                "p100": percentile(total_ms, 100),
            }
        },
        "budget_pass": percentile(total_ms, 50) <= 200
    }
