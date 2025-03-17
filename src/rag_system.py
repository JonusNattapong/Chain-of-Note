"""
Main RAG system using Chain-of-Note technique.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .data_loader import DocumentLoader
from .embeddings import EmbeddingModel, SentenceTransformerEmbeddings
from .document_store import DocumentStore
from .chain_of_note import ChainOfNote

class ChainOfNoteRAG:
    """Complete RAG system with Chain-of-Note."""
    
    def __init__(
        self, 
        embedding_model: Optional[EmbeddingModel] = None,
        llm_model_name: str = "google/flan-t5-large"
    ):
        """Initialize the RAG system.
        
        Args:
            embedding_model: Model for generating document embeddings
            llm_model_name: Name of the language model for generation
        """
        self.embedding_model = embedding_model or SentenceTransformerEmbeddings()
        self.document_store = DocumentStore()
        self.chain_of_note = ChainOfNote(model_name=llm_model_name)
        
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the RAG system.
        
        Args:
            documents: List of document dictionaries with "content" field
        """
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.embed_documents(contents)
        self.document_store.add_documents(documents, embeddings)
        
    def process_documents_from_loader(self, loader: DocumentLoader, chunk_size: int = 500) -> None:
        """Process documents from a document loader.
        
        Args:
            loader: DocumentLoader containing documents
            chunk_size: Size of document chunks
        """
        chunks = loader.create_chunks(chunk_size=chunk_size)
        self.add_documents(chunks)
        
    def query(
        self, 
        query: str, 
        top_k: int = 5, 
        return_context: bool = False,
        return_notes: bool = False
    ) -> Dict[str, Any]:
        """Process a query through the RAG system.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_context: Whether to include retrieved documents in response
            return_notes: Whether to include generated notes in response
            
        Returns:
            Response with answer and optionally context and notes
        """
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.document_store.search(query_embedding, top_k=top_k)
        
        # Generate response using Chain-of-Note
        response = self.chain_of_note.generate_response(
            query=query,
            documents=retrieved_docs,
            return_notes=return_notes
        )
        
        # Add retrieved context if requested
        if return_context:
            response["context"] = retrieved_docs
            
        return response
