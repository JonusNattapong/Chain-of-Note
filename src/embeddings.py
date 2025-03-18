"""
Embedding models for semantic document retrieval.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral client if API key is available
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class EmbeddingModel:
    """Base class for embedding models."""
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings as numpy array
        """
        raise NotImplementedError
        
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        raise NotImplementedError

class SentenceTransformerEmbeddings(EmbeddingModel):
    """Embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables. Please set it in .env file.")
        self.model = SentenceTransformer(model_name, token=token)
        
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents using Sentence Transformer.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings
        """
        return self.model.encode(documents, convert_to_numpy=True)
        
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using Sentence Transformer.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.model.encode([query], convert_to_numpy=True)[0]
        
class HuggingFaceEmbeddings(EmbeddingModel):
    """Custom embeddings using Hugging Face models."""
    
    def __init__(self, model_name: str = "google/bert-base-uncased"):
        """Initialize with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to generate a single vector per document."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents using Hugging Face model.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings
        """
        encoded_input = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        return embeddings.cpu().numpy()
        
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using Hugging Face model.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_documents([query])[0]

class MistralEmbeddings(EmbeddingModel):
    """Embeddings using Mistral AI API."""
    
    def __init__(self):
        """Initialize with Mistral AI client."""
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in .env file.")
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents using Mistral AI.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings
        """
        embeddings = []
        for doc in documents:
            response = self.client.embeddings(
                model="mistral-embed",
                input=doc
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
        
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using Mistral AI.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_documents([query])[0]
