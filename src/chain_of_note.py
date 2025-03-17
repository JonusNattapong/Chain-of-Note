"""
Implementation of Chain-of-Note technique for improved RAG.
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ChainOfNote:
    """Implements the Chain-of-Note technique for improved RAG."""
    
    def __init__(self, model_name: str = "google/flan-t5-large"):
        """Initialize with a language model.
        
        Args:
            model_name: Name of the language model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name) if "gpt" in model_name.lower() else None
        
        # For T5 and other encoder-decoder models
        if self.model is None:
            self.generator = pipeline(
                "text2text-generation", 
                model=model_name, 
                tokenizer=self.tokenizer, 
                device=0 if self.device == "cuda" else -1
            )
            
    def _create_note_prompt(self, query: str, documents: List[Dict]) -> str:
        """Create a prompt for note generation.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Formatted prompt for note generation
        """
        prompt = f"Question: {query}\n\nRelevant Context:\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            prompt += f"[Document {i}]: {content}\n\n"
            
        prompt += "Based on the above context, generate detailed notes that will help answer the question. Include specific facts, dates, numbers, and quotes from the documents. Identify any contradictions or gaps in the information provided.\n\nNotes:"
        
        return prompt
        
    def _create_answer_prompt(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Create a prompt for answer generation.
        
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
            content = doc["content"][:300]  # Truncate if too long
            prompt += f"[Document {i}]: {content}...\n\n"
            
        prompt += f"Notes on the context:\n{notes}\n\n"
        prompt += "Using the provided context documents and notes, please answer the question. Only include information that is supported by the context. If the context doesn't provide enough information, acknowledge the limitations in your answer.\n\nAnswer:"
        
        return prompt
        
    def generate_note(self, query: str, documents: List[Dict]) -> str:
        """Generate notes based on query and documents.
        
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
                temperature=0.3
            )
            note = result[0]["generated_text"]
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.3,
                do_sample=True
            )
            note = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            
        return note.strip()
        
    def generate_answer(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Generate final answer based on query, notes, and documents.
        
        Args:
            query: User query
            notes: Generated notes
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        prompt = self._create_answer_prompt(query, notes, documents)
        
        if self.model is None:  # For T5 and similar models
            result = self.generator(
                prompt, 
                max_length=512,
                num_return_sequences=1,
                temperature=0.3
            )
            answer = result[0]["generated_text"]
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.3,
                do_sample=True
            )
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            
        return answer.strip()
        
    def generate_response(self, query: str, documents: List[Dict], return_notes: bool = False) -> Dict[str, Any]:
        """Generate a complete response using Chain-of-Note.
        
        Args:
            query: User query
            documents: Retrieved documents
            return_notes: Whether to include notes in the response
            
        Returns:
            Dictionary with answer and optionally notes
        """
        notes = self.generate_note(query, documents)
        answer = self.generate_answer(query, notes, documents)
        
        response = {"answer": answer}
        
        if return_notes:
            response["notes"] = notes
            
        return response
