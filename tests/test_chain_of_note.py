"""
Tests for Chain-of-Note functionality.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chain_of_note import ChainOfNote

class TestChainOfNote(unittest.TestCase):
    """Test cases for Chain-of-Note."""
    
    def setUp(self):
        """Set up test data."""
        self.test_query = "What is artificial intelligence?"
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
        
        self.mock_note = "AI is intelligence demonstrated by machines. Machine learning is a subset of AI."
    
    @patch('src.chain_of_note.pipeline')
    @patch('src.chain_of_note.AutoTokenizer')
    def test_prompt_creation(self, mock_tokenizer, mock_pipeline):
        """Test prompt creation methods."""
        # Set up mock
        mock_generator = MagicMock()
        mock_pipeline.return_value = mock_generator
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Create ChainOfNote instance
        chain = ChainOfNote(model_name="test-model")
        
        # Test note prompt
        note_prompt = chain._create_note_prompt(self.test_query, self.test_documents)
        self.assertIn(self.test_query, note_prompt)
        self.assertIn("Document 1", note_prompt)
        self.assertIn("Document 2", note_prompt)
        
        # Test answer prompt
        answer_prompt = chain._create_answer_prompt(self.test_query, self.mock_note, self.test_documents)
        self.assertIn(self.test_query, answer_prompt)
        self.assertIn(self.mock_note, answer_prompt)
    
    @patch('src.chain_of_note.pipeline')
    @patch('src.chain_of_note.AutoTokenizer')
    def test_generate_response(self, mock_tokenizer, mock_pipeline):
        """Test generating a complete response."""
        # Set up mock
        mock_generator = MagicMock()
        mock_generator.return_value = [{"generated_text": "This is a test note."}]
        mock_pipeline.return_value = mock_generator
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Create ChainOfNote instance
        chain = ChainOfNote(model_name="test-model")
        
        # Mock generate_note and generate_answer methods
        chain.generate_note = MagicMock(return_value="Test note")
        chain.generate_answer = MagicMock(return_value="Test answer")
        
        # Test generate_response method
        response = chain.generate_response(
            query=self.test_query, 
            documents=self.test_documents, 
            return_notes=True
        )
        
        # Assertions
        self.assertEqual(response["answer"], "Test answer")
        self.assertEqual(response["notes"], "Test note")
        chain.generate_note.assert_called_once()
        chain.generate_answer.assert_called_once()

if __name__ == "__main__":
    unittest.main()
