"""
Main RAG system using Chain-of-Note technique.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .data_loader import DocumentLoader
from .embeddings import EmbeddingModel, SentenceTransformerEmbeddings, MistralEmbeddings
from .document_store import DocumentStore
from .chain_of_note import ChainOfNote, MistralAIChat


class ChainOfNoteRAG:
    """RAG system using Chain-of-Note."""

    def __init__(self, embedding_model_name="sentence-transformers/all-mpnet-base-v2", use_mistral_embeddings=False, llm_model_name="google/flan-t5-large", use_mistral_llm=False, chain_of_note=None):
        """Initialize with embedding model and Chain-of-Note instance.

        Args:
            embedding_model_name: Name of the embedding model to use.
            use_mistral_embeddings: Whether to use Mistral AI for embeddings.
            llm_model_name: Name of the language model to use for Chain-of-Note.
            use_mistral_llm: Whether to use Mistral AI for the Chain-of-Note language model.
            chain_of_note: Optional pre-configured ChainOfNote instance.
        """
        if use_mistral_embeddings:
            self.embedding_model = MistralEmbeddings()
        else:
            self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

        self.document_store = DocumentStore()

        if chain_of_note:
            self.chain_of_note = chain_of_note
        else:
            self.chain_of_note = ChainOfNote(model_name=llm_model_name, use_mistral=use_mistral_llm)


    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the RAG system."""
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.embed_documents(contents)
        self.document_store.add_documents(documents, embeddings)

    def process_documents_from_loader(self, loader: DocumentLoader, chunk_size: int = 500) -> None:
        """Process documents from a document loader."""
        chunks = loader.create_chunks(chunk_size=chunk_size)
        self.add_documents(chunks)

    def query(
        self,
        query: str,
        top_k: int = 5,
        return_context: bool = False,
        return_notes: bool = False
    ) -> Dict[str, Any]:
        """Process a query through the RAG system."""
        query_embedding = self.embedding_model.embed_query(query)
        retrieved_docs = self.document_store.search(query_embedding, top_k=top_k)
        response = self.chain_of_note.generate_response(
            query=query,
            documents=retrieved_docs,
            return_notes=return_notes
        )

        if return_context:
            response["context"] = retrieved_docs

        return response
