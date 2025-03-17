"""
Tests for document store functionality.
"""
import os
import sys
import unittest
import numpy as np
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_store import DocumentStore

class TestDocumentStore(unittest.TestCase):
    """Test cases for document store."""
    
    def setUp(self):
        """Set up document store for testing."""
        self.document_store = DocumentStore(embedding_dim=128)
        self.test_docs = [
            {"content": "Test document one", "metadata": {"source": "test"}},
            {"content": "Test document two", "metadata": {"source": "test"}},
            {"content": "Test document three", "metadata": {"source": "test"}}
        ]
        self.test_embeddings = np.random.rand(3, 128).astype(np.float32)
    
    def test_add_documents(self):
        """Test adding documents to the store."""
        self.document_store.add_documents(self.test_docs, self.test_embeddings)
        self.assertEqual(len(self.document_store.documents), 3)
        self.assertEqual(self.document_store.index.ntotal, 3)
    
    def test_search(self):
        """Test searching for documents."""
        self.document_store.add_documents(self.test_docs, self.test_embeddings)
        
        # Create a test query embedding
        query_embedding = np.random.rand(1, 128).astype(np.float32)
        
        # Test search with various k values
        for k in [1, 2, 3]:
            results = self.document_store.search(query_embedding, top_k=k)
            self.assertEqual(len(results), k)
            for result in results:
                self.assertTrue("score" in result)
                self.assertTrue("content" in result)
                self.assertTrue("metadata" in result)
    
    def test_save_load(self):
        """Test saving and loading the document store."""
        # Add documents
        self.document_store.add_documents(self.test_docs, self.test_embeddings)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.faiss') as temp:
            index_path = temp.name
            self.document_store.save(index_path)
            
            # Create a new document store and load the index
            new_store = DocumentStore(embedding_dim=128)
            new_store.documents = self.document_store.documents  # We need to manually copy the documents
            new_store.load(index_path)
            
            # Check that the index was loaded properly
            self.assertEqual(new_store.index.ntotal, 3)
            
            # Test search on the loaded index
            query_embedding = np.random.rand(1, 128).astype(np.float32)
            results = new_store.search(query_embedding, top_k=2)
            self.assertEqual(len(results), 2)

if __name__ == "__main__":
    unittest.main()
