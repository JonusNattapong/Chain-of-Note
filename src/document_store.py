"""
Vector store for document retrieval.
"""
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple

class DocumentStore:
    """Vector database for storing and retrieving documents."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize the document store.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries
            embeddings: Document embeddings
        """
        if self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with similarity scores
        """
        if self.index.ntotal == 0:
            return []
            
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
                
        return results
        
    def save(self, file_path: str) -> None:
        """Save the index to disk.
        
        Args:
            file_path: Path to save the index
        """
        faiss.write_index(self.index, file_path)
        
    def load(self, file_path: str) -> None:
        """Load the index from disk.
        
        Args:
            file_path: Path to the saved index
        """
        self.index = faiss.read_index(file_path)
