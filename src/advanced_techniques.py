"""
Advanced Chain-of-Note techniques to further improve RAG performance.
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .chain_of_note import ChainOfNote

class EnhancedChainOfNote(ChainOfNote):
    """Enhanced Chain-of-Note with additional techniques."""
    
    def __init__(self, 
                 model_name: str = "google/flan-t5-large", 
                 use_self_critique: bool = True,
                 use_source_ranking: bool = True,
                 use_claim_verification: bool = False,
                 temperature: float = 0.3):
        """Initialize with a language model and advanced options.
        
        Args:
            model_name: Name of the language model to use
            use_self_critique: Whether to use self-critique
            use_source_ranking: Whether to rank sources
            use_claim_verification: Whether to verify claims
            temperature: Temperature for generation
        """
        super().__init__(model_name)
        self.use_self_critique = use_self_critique
        self.use_source_ranking = use_source_ranking
        self.use_claim_verification = use_claim_verification
        self.temperature = temperature
        
    def _create_note_prompt(self, query: str, documents: List[Dict]) -> str:
        """Create an enhanced prompt for note generation.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Formatted prompt for note generation
        """
        prompt = f"Question: {query}\n\nRelevant Context:\n"
        
        # Optionally rank sources by relevance
        if self.use_source_ranking:
            # Simple heuristic: use score if available
            if all("score" in doc for doc in documents):
                documents = sorted(documents, key=lambda x: x["score"])
        
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            source = doc["metadata"].get("source", f"Document {i}")
            prompt += f"[Source {i}: {source}]: {content}\n\n"
            
        prompt += "Based on the above context, generate detailed notes that will help answer the question. "
        prompt += "Include specific facts, dates, numbers, and quotes from the sources. "
        
        if self.use_source_ranking:
            prompt += "Prioritize information from the most relevant sources. "
            
        if self.use_claim_verification:
            prompt += "For each claim, identify which source supports it. "
            prompt += "If sources contradict each other, note the contradictions. "
            
        prompt += "\nNotes:"
        
        return prompt
        
    def _create_answer_prompt(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Create an enhanced prompt for answer generation.
        
        Args:
            query: User query
            notes: Generated notes
            documents: Retrieved documents
            
        Returns:
            Formatted prompt for answer generation
        """
        prompt = f"Question: {query}\n\n"
        
        prompt += "Context Documents:\n"
        for i, doc in enumerate(documents, 1):
            content = doc["content"][:100]  # Brief summary only
            source = doc["metadata"].get("source", f"Document {i}")
            prompt += f"[Source {i}: {source}]: {content}...\n"
            
        prompt += f"\nDetailed Notes on the context:\n{notes}\n\n"
        prompt += "Using the provided context documents and notes, please answer the question. "
        prompt += "Only include information that is supported by the context. "
        
        if self.use_self_critique:
            prompt += "First, provide your answer. "
            prompt += "Then, critically assess your answer for potential inaccuracies or missing information. "
            prompt += "Finally, provide an improved answer that addresses any issues identified. "
        
        prompt += "\nAnswer:"
        
        return prompt
        
    def generate_note(self, query: str, documents: List[Dict]) -> str:
        """Generate enhanced notes with source attribution.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Generated notes
        """
        prompt = self._create_note_prompt(query, documents)
        
        if self.model is None:  # For T5 and similar models
            result = self.generator(
                prompt, 
                max_length=512,
                num_return_sequences=1,
                temperature=self.temperature
            )
            note = result[0]["generated_text"]
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=self.temperature,
                do_sample=True
            )
            note = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            
        return note.strip()
        
    def generate_answer_with_critique(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Generate final answer with self-critique if enabled.
        
        Args:
            query: User query
            notes: Generated notes
            documents: Retrieved documents
            
        Returns:
            Generated answer with critique if enabled
        """
        prompt = self._create_answer_prompt(query, notes, documents)
        
        if self.model is None:  # For T5 and similar models
            result = self.generator(
                prompt, 
                max_length=512,
                num_return_sequences=1,
                temperature=self.temperature
            )
            answer = result[0]["generated_text"]
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=self.temperature,
                do_sample=True
            )
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            
        return answer.strip()
    
    def verify_claims(self, answer: str, documents: List[Dict]) -> Dict[str, Any]:
        """Verify claims in the answer against source documents.
        
        Args:
            answer: Generated answer
            documents: Source documents
            
        Returns:
            Dictionary with verification results
        """
        if not self.use_claim_verification:
            return {"verified": True}
            
        # Create a verification prompt
        prompt = "I need to verify the following claims against the provided sources:\n\n"
        prompt += f"Claims to verify: {answer}\n\n"
        prompt += "Sources:\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            source = doc["metadata"].get("source", f"Document {i}")
            prompt += f"[Source {i}: {source}]: {content}\n\n"
            
        prompt += "For each claim in the text, indicate whether it is:\n"
        prompt += "1. SUPPORTED: Directly supported by the sources\n"
        prompt += "2. PARTIALLY SUPPORTED: Some aspects are supported, but details may differ\n"
        prompt += "3. UNSUPPORTED: Cannot be verified from the given sources\n"
        prompt += "4. CONTRADICTED: Directly contradicted by the sources\n\n"
        prompt += "Provide your verification results in a structured format.\n\n"
        prompt += "Verification results:"
        
        # Get verification results
        if self.model is None:  # For T5 and similar models
            result = self.generator(
                prompt, 
                max_length=512,
                num_return_sequences=1,
                temperature=0.2  # Lower temperature for verification
            )
            verification = result[0]["generated_text"]
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.2,
                do_sample=True
            )
            verification = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            
        return {
            "verified": False,  # Set to False to indicate verification was performed
            "verification_results": verification.strip()
        }
        
    def generate_response(self, query: str, documents: List[Dict], return_notes: bool = False, return_verification: bool = False) -> Dict[str, Any]:
        """Generate a complete response using Enhanced Chain-of-Note.
        
        Args:
            query: User query
            documents: Retrieved documents
            return_notes: Whether to include notes in the response
            return_verification: Whether to include claim verification
            
        Returns:
            Dictionary with answer and optional components
        """
        # Generate notes
        notes = self.generate_note(query, documents)
        
        # Generate answer with self-critique if enabled
        answer = self.generate_answer_with_critique(query, notes, documents)
        
        response = {"answer": answer}
        
        if return_notes:
            response["notes"] = notes
            
        # Perform claim verification if requested
        if return_verification and self.use_claim_verification:
            verification = self.verify_claims(answer, documents)
            response["verification"] = verification
            
        return response
