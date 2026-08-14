import os
import subprocess
import time
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def run_command(cmd: str):
    print(f"Running: {cmd}")
    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    return time.time() - start

def load_data(parquet_path, num_records):
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    return df.head(num_records)

def evaluate_retrieval(index_file, mapping_file, queries, model):
    print("Loading index and mapping...")
    index = faiss.read_index(index_file)
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    print(f"Encoding {len(queries)} queries...")
    start = time.time()
    query_embeddings = model.encode(queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    query_embeddings = np.array(query_embeddings).astype("float32")
    encoding_time = time.time() - start
    
    print("Searching FAISS index...")
    start = time.time()
    k = 5
    distances, indices = index.search(query_embeddings, k)
    search_time = time.time() - start
    
    recall_at_1 = 0
    recall_at_5 = 0
    mrr = 0.0
    
    for i, query in enumerate(queries):
        retrieved_ids = indices[i]
        found_rank = None
        for rank, retrieved_idx in enumerate(retrieved_ids):
            if retrieved_idx == -1:
                continue
            chunk_data = mapping[str(retrieved_idx)]
            # We use source_query to match relevance
            if chunk_data["metadata"]["source_query"] == query:
                found_rank = rank + 1
                break
                
        if found_rank is not None:
            recall_at_5 += 1
            if found_rank == 1:
                recall_at_1 += 1
            mrr += 1.0 / found_rank
            
    num_q = len(queries)
    metrics = {
        "Recall@1": recall_at_1 / num_q,
        "Recall@5": recall_at_5 / num_q,
        "MRR": mrr / num_q,
        "Encoding_Latency_sec": encoding_time,
        "Search_Latency_sec": search_time
    }
    return metrics

def main():
    strategies = ["fixed", "semantic", "metadata"]
    num_records = 200
    model_name = "BAAI/bge-small-en-v1.5"
    
    print("Loading embedding model for evaluation...")
    model = SentenceTransformer(model_name)
    
    parquet_path = "../data/hinval.parquet"
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*40}")
        print(f"Benchmarking Strategy: {strategy.upper()}")
        print(f"{'='*40}")
        
        chunks_file = f"../data/chunks_{strategy}.json"
        index_file = f"../data/vector_{strategy}.index"
        mapping_file = f"../data/mapping_{strategy}.json"
        
        import sys
        # 1. Preprocess
        cmd_prep = f'"{sys.executable}" preprocess.py --num_records {num_records} --strategy {strategy} --output_file "{chunks_file}"'
        prep_time = run_command(cmd_prep)
        
        # We load queries now that preprocess might have downloaded the parquet file
        if "queries" not in locals():
            if not os.path.exists(parquet_path):
                print(f"Error: {parquet_path} not found even after preprocess.")
                return
            df = load_data(parquet_path, num_records)
            queries = df["query"].tolist()
        
        # 2. Embed
        cmd_embed = f'"{sys.executable}" embed.py --chunks_file "{chunks_file}" --index_file "{index_file}" --mapping_file "{mapping_file}"'
        embed_time = run_command(cmd_embed)
        
        # 3. Evaluate
        metrics = evaluate_retrieval(index_file, mapping_file, queries, model)
        
        results[strategy] = {
            "Prep_Time": prep_time,
            "Embed_Time": embed_time,
            **metrics
        }
        
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    # Format as table
    print(f"{'Strategy':<12} | {'R@1':<6} | {'R@5':<6} | {'MRR':<6} | {'Prep(s)':<8} | {'Embed(s)':<8}")
    print("-" * 60)
    for s in strategies:
        r = results[s]
        print(f"{s:<12} | {r['Recall@1']:.4f} | {r['Recall@5']:.4f} | {r['MRR']:.4f} | {r['Prep_Time']:.2f}    | {r['Embed_Time']:.2f}")

if __name__ == "__main__":
    main()
