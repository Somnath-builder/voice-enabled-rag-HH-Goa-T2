import json
import argparse
import os
import requests
import pandas as pd
from typing import List, Dict, Any

# Simple Fixed-Size Chunker
def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Preprocess and chunk MSMARCO-XI dataset.")
    parser.add_argument("--num_records", type=int, default=100, help="Number of records to process.")
    parser.add_argument("--output_file", type=str, default="../data/chunks.json", help="Path to output chunks.")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    parquet_path = "../data/hinval.parquet"
    if not os.path.exists(parquet_path):
        print("Downloading hinval.parquet directly...")
        url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(parquet_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024 * 1024 * 10) < 8192: # Print every ~10MB
                        print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                
    print(f"Reading {args.num_records} records from parquet file...")
    
    # Read with pandas (which uses pyarrow backend but often handles nested structures better locally)
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.head(args.num_records)
    
    all_chunks = []
    chunk_id_counter = 0
    
    for i, row in df.iterrows():
        query = row.get("query", "")
        passages = row.get("passages", {})
        
        passage_texts = []
        if isinstance(passages, dict):
            # Try to get English passages first as our embedding model is BGE-small-en
            if "English_passages" in passages:
                passage_texts = passages["English_passages"]
            elif "Translated_passages" in passages:
                passage_texts = passages["Translated_passages"]
            
            # In pandas, this might be a numpy array, convert to list
            if hasattr(passage_texts, "tolist"):
                passage_texts = passage_texts.tolist()
        elif isinstance(passages, list):
            # Fallback just in case
            passage_texts = [str(p) for p in passages]
        
        for p_text in passage_texts:
            if not isinstance(p_text, str) or not p_text.strip():
                continue
            
            text_chunks = fixed_size_chunking(p_text, chunk_size=500, overlap=50)
            
            for chunk_text in text_chunks:
                all_chunks.append({
                    "chunk_id": str(chunk_id_counter),
                    "text": chunk_text,
                    "metadata": {
                        "source_query": query,
                    }
                })
                chunk_id_counter += 1
                
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} queries, extracted {chunk_id_counter} chunks...")
            
    print(f"Finished processing. Total chunks extracted: {len(all_chunks)}")
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Chunks saved to {args.output_file}")

if __name__ == "__main__":
    main()
