import json
import argparse
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def main():
    parser = argparse.ArgumentParser(description="Embed chunks and build FAISS index.")
    parser.add_argument("--chunks_file", type=str, default="../data/chunks.json", help="Path to chunks.json.")
    parser.add_argument("--index_file", type=str, default="../data/vector.index", help="Path to save FAISS index.")
    parser.add_argument("--mapping_file", type=str, default="../data/chunk_mapping.json", help="Path to save chunk mappings.")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.chunks_file):
        print(f"Error: {args.chunks_file} not found.")
        return
        
    print(f"Loading chunks from {args.chunks_file}...")
    with open(args.chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    if not chunks:
        print("No chunks found to embed.")
        return
        
    print(f"Loaded {len(chunks)} chunks.")
    
    # Extract text
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Loading embedding model: {args.model}")
    start_time = time.time()
    # We use sentence-transformers. Using ONNX explicitly might require Optimum, 
    # but sentence-transformers natively is very fast for small models.
    model = SentenceTransformer(args.model)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    
    print("Generating embeddings (this may take a while)...")
    start_time = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings generated in {time.time() - start_time:.2f} seconds.")
    
    # FAISS expects float32
    embeddings = np.array(embeddings).astype("float32")
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # L2 distance on normalized vectors = Cosine similarity
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(args.index_file), exist_ok=True)
    
    print(f"Saving FAISS index to {args.index_file}...")
    faiss.write_index(index, args.index_file)
    
    print(f"Saving chunk mapping to {args.mapping_file}...")
    # Map index ID to chunk data for retrieval
    mapping = {i: chunk for i, chunk in enumerate(chunks)}
    with open(args.mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
        
    print("Indexing complete!")

if __name__ == "__main__":
    main()
