import requests
import time
import numpy as np
import json
import random

# Use a subset of known queries to test the API
QUERIES = [
    "What is a corporation?",
    "How does a cell divide?",
    "Who is Rachel Carson?",
    "When was the Declaration of Independence signed?",
    "What are the benefits of a B Corp?",
    "What is the theory of relativity?",
    "What are the symptoms of COVID-19?",
    "How do black holes form?",
    "What is the capital of India?",
    "Why is the sky blue?",
    "What is photosynthesis?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the largest mammal?",
    "What is artificial intelligence?",
    "How does a car engine work?",
    "What is the difference between DNA and RNA?",
    "Who painted the Mona Lisa?",
    "What is the deepest ocean?",
    "How do vaccines work?",
    "What is quantum computing?",
    "What is the tallest mountain?",
    "Who invented the telephone?",
    "What is the powerhouse of the cell?",
    "What is climate change?",
    "How many bones are in the human body?",
    "What is the largest planet in our solar system?",
    "Who was the first person on the moon?",
    "What is the boiling point of water?",
    "What is the chemical formula for water?",
    "What is gravity?",
    "What is a black hole?",
    "Who discovered penicillin?",
    "What is the closest star to Earth?",
    "What is the human genome project?",
    "What is the freezing point of water?",
    "Who is the author of Harry Potter?",
    "What is the largest organ in the human body?",
    "What is the smallest country in the world?",
    "Who painted the Starry Night?",
    "What is the largest desert in the world?",
    "What is the hardest natural substance?",
    "What is the most abundant gas in Earth's atmosphere?",
    "Who developed the theory of evolution?",
    "What is the fastest animal on land?",
    "What is the largest ocean?",
    "What is the currency of Japan?",
    "Who wrote Hamlet?",
    "What is the capital of France?",
    "What is the primary language spoken in Brazil?"
]

API_URL = "http://localhost:8000/api/query"

def run_benchmark():
    print(f"Starting benchmark with {len(QUERIES)} queries...")
    
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    
    # Randomize order
    queries_to_run = list(QUERIES)
    random.shuffle(queries_to_run)
    
    successful_requests = 0
    
    for i, query in enumerate(queries_to_run):
        try:
            start_req = time.time()
            response = requests.post(API_URL, json={"query": query}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                retrieval_latencies.append(data.get("retrieval_latency_ms", 0))
                generation_latencies.append(data.get("generation_latency_ms", 0))
                total_latencies.append(data.get("total_latency_ms", 0))
                successful_requests += 1
            else:
                print(f"[{i+1}/{len(QUERIES)}] Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"[{i+1}/{len(QUERIES)}] Request threw an exception: {e}")
            
        print(f"[{i+1}/{len(QUERIES)}] Processed query: '{query[:30]}...'")
        
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Requests Sent: {len(QUERIES)}")
    print(f"Successful Requests: {successful_requests}")
    
    if successful_requests > 0:
        print("\nRetrieval Latency (ms):")
        print(f"  P50 (Median): {np.percentile(retrieval_latencies, 50):.2f} ms")
        print(f"  P70:          {np.percentile(retrieval_latencies, 70):.2f} ms")
        print(f"  P100 (Max):   {np.max(retrieval_latencies):.2f} ms")
        
        print("\nGeneration Latency (ms):")
        print(f"  P50 (Median): {np.percentile(generation_latencies, 50):.2f} ms")
        print(f"  P70:          {np.percentile(generation_latencies, 70):.2f} ms")
        print(f"  P100 (Max):   {np.max(generation_latencies):.2f} ms")
        
        print("\nTotal Backend API Latency (ms):")
        print(f"  P50 (Median): {np.percentile(total_latencies, 50):.2f} ms")
        print(f"  P70:          {np.percentile(total_latencies, 70):.2f} ms")
        print(f"  P100 (Max):   {np.max(total_latencies):.2f} ms")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
