"""
Document loading and preprocessing for RAG system.
"""
import os
import pandas as pd
from typing import List, Dict, Union, Optional
from datasets import Dataset

class DocumentLoader:
    """Handles document loading and preprocessing for RAG."""
    
    def __init__(self):
        """Initialize document loader."""
        self.documents = []
        
    def load_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Load a single text document.
        
        Args:
            text: The document text
            metadata: Optional metadata about the document
        """
        doc = {
            "content": text,
            "metadata": metadata or {}
        }
        self.documents.append(doc)
        
    def load_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Load multiple text documents.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        metadatas = metadatas or [{}] * len(texts)
        for text, metadata in zip(texts, metadatas):
            self.load_text(text, metadata)
            
    def load_csv(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]] = None) -> None:
        """Load documents from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            text_column: Column containing document text
            metadata_columns: Columns to include as metadata
        """
        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        
        metadatas = []
        if metadata_columns:
            for _, row in df.iterrows():
                metadata = {col: row[col] for col in metadata_columns if col in df.columns}
                metadatas.append(metadata)
        
        self.load_texts(texts, metadatas)
        
    def create_chunks(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
        """Split documents into chunks of specified size.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        for doc in self.documents:
            text = doc["content"]
            metadata = doc["metadata"]
            
            if len(text) <= chunk_size:
                chunks.append({"content": text, "metadata": metadata})
                continue
                
            # Split into overlapping chunks
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) < chunk_size / 2 and i > 0:  # Skip very small final chunks
                    continue
                
                chunk = {
                    "content": chunk_text,
                    "metadata": {**metadata, "chunk_id": len(chunks)}
                }
                chunks.append(chunk)
                
        return chunks
    
    def get_dataset(self) -> Dataset:
        """Convert documents to Hugging Face dataset.
        
        Returns:
            Dataset object with documents
        """
        contents = [doc["content"] for doc in self.documents]
        metadatas = [doc["metadata"] for doc in self.documents]
        
        dataset_dict = {
            "content": contents,
            "metadata": metadatas
        }
        
        return Dataset.from_dict(dataset_dict)
