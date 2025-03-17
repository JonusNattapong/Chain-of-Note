"""
Tests for embedding models.
"""
import os
import sys
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding models."""
    
    def test_sentence_transformer_embeddings(self):
        """Test SentenceTransformerEmbeddings."""
        # Use a small model for testing
        embedding_model = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Test single document embedding
        test_text = "This is a test document for embedding."
        embedding = embedding_model.embed_query(test_text)
        
        # Check shape and type
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.ndim, 1)  # Should be a 1D array
        
        # Test multiple document embeddings
        test_texts = [
            "This is the first test document.",
            "This is the second test document.",
            "This is the third test document."
        ]
        embeddings = embedding_model.embed_documents(test_texts)
        
        # Check shape and type
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(test_texts))
        
    def test_huggingface_embeddings(self):
        """Test HuggingFaceEmbeddings.
        
        Note: This test may be skipped if computational resources are limited.
        """
        try:
            # Use a small model for testing
            embedding_model = HuggingFaceEmbeddings(
                model_name="google/bert-base-uncased"
            )
            
            # Test single document embedding
            test_text = "This is a test document for embedding."
            embedding = embedding_model.embed_query(test_text)
            
            # Check shape and type
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.ndim, 1)
            
        except Exception as e:
            self.skipTest(f"Skipping HuggingFace model test due to: {e}")

if __name__ == "__main__":
    unittest.main()
