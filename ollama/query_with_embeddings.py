import json
import numpy as np
import requests
from typing import List, Dict
import os
import sys

class DocumentQuery:
    def __init__(self, embeddings_path: str, model_name="deepseek-r1:7b"):
        """Initialize with pre-computed embeddings"""
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        
        # Optimized parameters for better performance
        self.model_params = {
            "num_gpu": 1,
            "num_thread": 8,
            "batch_size": 2048,
            "num_ctx": 8192,
            "num_batch": 2048,
            "num_gqa": 8,
            "num_gpu_layers": -1,
            "f16_kv": True,
            "mmap": True,
            "gpu_layers": "all",
            "rope_scaling": {"type": "linear", "factor": 4.0}
        }
        
        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            self.embeddings_data = json.load(f)
        print(f"Loaded {len(self.embeddings_data)} embeddings")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using Ollama"""
        print("Getting query embedding...", end="", flush=True)
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            headers=self.headers,
            json={
                "model": self.model_name,
                "prompt": query,
                "options": self.model_params
            }
        )
        print(" Done")
        return response.json()["embedding"]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def query(self, question: str, top_k: int = 3) -> None:
        """Query the document using pre-computed embeddings"""
        # Get query embedding
        query_embedding = self._get_query_embedding(question)
        
        print("Finding relevant sections...", end="", flush=True)
        # Find most similar chunks
        similarities = [
            (
                self._cosine_similarity(query_embedding, chunk["embedding"]),
                chunk["content"]
            )
            for chunk in self.embeddings_data
        ]
        
        # Sort by similarity and get top k
        similarities.sort(reverse=True)
        relevant_chunks = [content for _, content in similarities[:top_k]]
        print(" Done")
        
        # Format prompt for better response
        context = "\n\n---\n\n".join(relevant_chunks)
        prompt = f"""Given this technical document context about a motor control module:

{context}

Question: {question}

Please provide a detailed technical answer based on the context above. If the information is not directly available in the context, please indicate that."""

        print("\nGenerating answer (streaming):")
        print("-" * 80)
        
        try:
            # Stream the response for faster feedback
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": self.model_params
                },
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        print(json_response['response'], end='', flush=True)
            print("\n" + "-" * 80)
            
        except Exception as e:
            print(f"\nError generating response: {str(e)}")
            response = getattr(e, 'response', None)
            if response is not None:
                print(f"Response content: {response.text}")
            raise

if __name__ == "__main__":
    # Get embeddings file path
    print("\nAvailable embedding files:")
    for file in os.listdir():
        if file.endswith("_embeddings.json"):
            print(f"- {file}")
    
    embeddings_path = input("\nEnter the path to your embeddings file: ")
    
    try:
        # Initialize query system
        query_system = DocumentQuery(embeddings_path)
        
        # Interactive query loop
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            try:
                query_system.query(question)
            except Exception as e:
                print(f"\nError during query: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
