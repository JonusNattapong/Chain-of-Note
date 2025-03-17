"""
Tests for the complete RAG system.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import ChainOfNoteRAG
from src.data_loader import DocumentLoader

class TestRAGSystem(unittest.TestCase):
    """Test cases for RAG system."""
    
    def setUp(self):
        """Set up test data."""
        self.test_documents = [
            {
                "content": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                "metadata": {"source": "Wikipedia", "topic": "AI"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"source": "Textbook", "topic": "ML"}
            }
        ]
        
        self.test_query = "What is artificial intelligence?"
        
    def test_add_documents(self):
        """Test adding documents to the RAG system."""
        # Mock dependencies
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = MagicMock()
        
        mock_document_store = MagicMock()
        
        # Create RAG system with mocked components
        rag_system = ChainOfNoteRAG(embedding_model=mock_embedding_model)
        rag_system.document_store = mock_document_store
        
        # Add documents
        rag_system.add_documents(self.test_documents)
        
        # Assertions
        mock_embedding_model.embed_documents.assert_called_once()
        mock_document_store.add_documents.assert_called_once()
    
    def test_process_documents_from_loader(self):
        """Test processing documents from a loader."""
        # Create a document loader with test data
        loader = DocumentLoader()
        for doc in self.test_documents:
            loader.load_text(doc["content"], doc["metadata"])
        
        # Mock RAG system
        rag_system = MagicMock()
        
        # Store original method
        original_method = ChainOfNoteRAG.process_documents_from_loader
        
        try:
            # Replace method with our test implementation
            ChainOfNoteRAG.process_documents_from_loader = MagicMock()
            
            # Call the method
            ChainOfNoteRAG.process_documents_from_loader(rag_system, loader, chunk_size=300)
            
            # Assertion
            ChainOfNoteRAG.process_documents_from_loader.assert_called_once()
            
        finally:
            # Restore original method
            ChainOfNoteRAG.process_documents_from_loader = original_method
    
    def test_query(self):
        """Test querying the RAG system."""
        # Mock dependencies
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = MagicMock()
        
        mock_document_store = MagicMock()
        mock_document_store.search.return_value = self.test_documents
        
        mock_chain_of_note = MagicMock()
        mock_chain_of_note.generate_response.return_value = {
            "answer": "Test answer",
            "notes": "Test notes"
        }
        
        # Create RAG system with mocked components
        rag_system = ChainOfNoteRAG(embedding_model=mock_embedding_model)
        rag_system.document_store = mock_document_store
        rag_system.chain_of_note = mock_chain_of_note
        
        # Query the system
        response = rag_system.query(
            query=self.test_query,
            top_k=3,
            return_context=True,
            return_notes=True
        )
        
        # Assertions
        mock_embedding_model.embed_query.assert_called_once_with(self.test_query)
        mock_document_store.search.assert_called_once()
        mock_chain_of_note.generate_response.assert_called_once()
        
        self.assertEqual(response["answer"], "Test answer")
        self.assertEqual(response["notes"], "Test notes")
        self.assertEqual(response["context"], self.test_documents)

if __name__ == "__main__":
    unittest.main()
