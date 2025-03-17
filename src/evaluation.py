"""
Evaluation metrics for RAG systems.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rouge_score import rouge_scorer
from transformers import pipeline
import torch

class RAGEvaluator:
    """Evaluator for RAG system outputs."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load model for relevance scoring if GPU is available
        self.relevance_model = None
        if torch.cuda.is_available():
            try:
                self.relevance_model = pipeline(
                    "text-classification",
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device=0
                )
            except Exception as e:
                print(f"Could not load relevance model: {e}")
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores between prediction and reference.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            Dictionary of ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    def calculate_relevance(self, query: str, response: str) -> Optional[float]:
        """Calculate relevance score between query and response.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Relevance score or None if model not available
        """
        if self.relevance_model is None:
            return None
        
        try:
            result = self.relevance_model(
                [query, response],
                truncation=True,
                padding=True,
                max_length=512
            )
            return result[0]["score"]
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return None
    
    def calculate_hallucination_score(self, answer: str, reference_docs: List[str]) -> float:
        """Calculate a simple hallucination score based on n-gram overlap.
        
        Lower score = less hallucination (better)
        
        Args:
            answer: Generated answer
            reference_docs: List of reference documents
            
        Returns:
            Hallucination score (0-1, lower is better)
        """
        # Create a single reference by concatenating all docs
        combined_reference = " ".join(reference_docs)
        
        # Get ROUGE-2 score (bigram overlap)
        rouge_scores = self.calculate_rouge(answer, combined_reference)
        
        # Turn into a hallucination score (1 - rouge2)
        # Lower score means less hallucination
        return 1.0 - rouge_scores["rouge2"]
    
    def evaluate_response(
        self, 
        query: str, 
        response: str,
        reference_docs: List[str],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of a RAG response.
        
        Args:
            query: User query
            response: Generated response
            reference_docs: Reference documents used
            reference_answer: Optional reference answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Calculate relevance if model is available
        relevance = self.calculate_relevance(query, response)
        if relevance is not None:
            results["relevance"] = relevance
        
        # Calculate hallucination score
        results["hallucination_score"] = self.calculate_hallucination_score(
            response, reference_docs
        )
        
        # Calculate ROUGE if reference answer is provided
        if reference_answer:
            results["rouge"] = self.calculate_rouge(response, reference_answer)
        
        return results
    
    def compare_systems(
        self,
        query: str,
        standard_rag_response: str,
        chain_of_note_response: str,
        reference_docs: List[str],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare standard RAG and Chain-of-Note RAG.
        
        Args:
            query: User query
            standard_rag_response: Response from standard RAG
            chain_of_note_response: Response from Chain-of-Note RAG
            reference_docs: Reference documents
            reference_answer: Optional reference answer
            
        Returns:
            Dictionary with comparison results
        """
        standard_eval = self.evaluate_response(
            query, standard_rag_response, reference_docs, reference_answer
        )
        
        chain_of_note_eval = self.evaluate_response(
            query, chain_of_note_response, reference_docs, reference_answer
        )
        
        return {
            "standard_rag": standard_eval,
            "chain_of_note_rag": chain_of_note_eval,
            "hallucination_reduction": standard_eval["hallucination_score"] - chain_of_note_eval["hallucination_score"]
        }
