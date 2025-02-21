import json
import requests
import numpy as np
from typing import List, Dict
import os

def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "deepseek-r1:7b", "prompt": text}
    )
    return response.json()["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_document(json_path: str, output_path: str):
    """Process document chunks and store their embeddings"""
    with open(json_path, 'r') as f:
        chunks = json.load(f)
    
    # Process in batches to avoid memory issues
    batch_size = 10
    embeddings_data = []
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            embedding = get_embedding(chunk["content"])
            embeddings_data.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "embedding": embedding
            })
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    
    # Save embeddings
    with open(output_path, 'w') as f:
        json.dump(embeddings_data, f)
    
    print(f"Saved embeddings to {output_path}")

def semantic_search(query: str, embeddings_path: str, top_k: int = 3) -> List[str]:
    """Search for most relevant chunks using semantic similarity"""
    # Load embeddings
    with open(embeddings_path, 'r') as f:
        embeddings_data = json.load(f)
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Calculate similarities
    similarities = [
        (cosine_similarity(query_embedding, chunk["embedding"]), chunk["content"])
        for chunk in embeddings_data
    ]
    
    # Sort by similarity and get top k
    similarities.sort(reverse=True)
    return [content for _, content in similarities[:top_k]]

if __name__ == "__main__":
    if not os.path.exists("error_reports_data/embeddings.json"):
        print("Creating embeddings for the document...")
        process_document(
            "error_reports_data/error_reports.json",
            "error_reports_data/embeddings.json"
        )
    
    while True:
        query = input("\nEnter your question about the error reports (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        print("\nSearching error reports...")
        relevant_chunks = semantic_search(query, "error_reports_data/embeddings.json")
        
        # Get answer from Ollama
        context = "\n\n---\n\n".join(relevant_chunks)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:7b",
                "prompt": f"Given this error report context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:",
                "stream": False
            }
        )
        
        print("\nAnswer:", response.json()["response"])
