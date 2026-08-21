"""Measure end-to-end full pipeline latency against the
200ms budget defined in the HH Goa 2026 requirements.

Usage:
    python -m scripts.benchmark_latency [n_queries]
"""
import statistics
import sys
import asyncio
import time
import os

# Add backend to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.orchestrator import orchestrator_service
from app.services.retriever import retriever_service

LATENCY_BUDGET_MS = 200

QUERIES = [
    "What is a corporation?",
    "How does tax law work?",
    "What are the rights of a shareholder?",
    "Explain intellectual property.",
    "What constitutes a breach of contract?",
    "What is the penalty for late filing?",
    "Define limited liability.",
    "What are the stages of arbitration?",
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

async def main_async():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("Warming up (model load + first inference)...")
    # Warmup the FAISS index by doing a dummy retrieval
    retriever_service.retrieve("warmup query")

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    
    print(f"Running {n} queries. This will call the Groq LLM API, so it may take a moment...")
    
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        
        # We process the query using the full orchestrated pipeline (Retrieval + JSON LLM)
        resp = await orchestrator_service.process_query(query)
        
        total_ms.append(resp["total_latency_ms"])
        retrieval_ms.append(resp["retrieval_latency_ms"])
        generation_ms.append(resp["generation_latency_ms"])
        
        print(f"Query {i+1}/{n} completed in {resp['total_latency_ms']:.2f}ms")
        
        # Sleep to avoid Groq free tier rate limits (30 RPM)
        if i < n - 1:
            await asyncio.sleep(2.1)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<15}{'avg':>8}{'p50':>8}{'p70':>8}{'p100':>8}   (ms)")
    for name, values in [("retrieval", retrieval_ms), ("generation", generation_ms), ("total", total_ms)]:
        print(
            f"{name:<15}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 70):>8.2f}"
            f"{percentile(values, 100):>8.2f}"
        )

    p50_total = percentile(total_ms, 50)
    p70_total = percentile(total_ms, 70)
    p100_total = percentile(total_ms, 100)
    
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms")
    print(f"P50 Total: {p50_total:.2f}ms")
    print(f"P70 Total: {p70_total:.2f}ms")
    print(f"P100 Total: {p100_total:.2f}ms")
    
    if p50_total <= LATENCY_BUDGET_MS:
        print("\nPASS: P50 is within budget")
    else:
        print("\nFAIL: P50 is over budget")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
