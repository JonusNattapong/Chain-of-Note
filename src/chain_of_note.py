"""
Implementation of Chain-of-Note technique for improved RAG.
"""
from typing import List, Dict, Any, Optional, Tuple
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mistralai.client import MistralClient
import os

class MistralAIChat:
    """Handles chat completion using Mistral AI API."""

    def __init__(self, model_name: str = "mistral-medium"):
        """Initializes Mistral AI client."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        self.client = MistralClient(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """Generates a response using Mistral AI chat completion."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        chat_response = self.client.chat(
            model=self.model_name,
            messages=messages
        )
        return chat_response.choices[0].message.content

class ChainOfNote:
    """Implements the Chain-of-Note technique for improved RAG."""
    
    def __init__(self, model_name: str = "google/flan-t5-large", use_mistral: bool = False):
        """Initialize with a language model.

        Args:
            model_name: Name of the language model to use.
            use_mistral: Whether to use Mistral AI for generation.
        """
        self.model_name = model_name
        self.use_mistral = use_mistral
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.use_mistral:
            self.mistral_chat = MistralAIChat(model_name=model_name)
        else:
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
        """Create a prompt for note generation."""
        prompt = f"Question: {query}\n\nRelevant Context:\n"
        for i, doc in enumerate(documents, 1):
            prompt += f"[Document {i}]: {doc['content']}\n\n"
        prompt += "Based on the above context, generate detailed notes that will help answer the question. Include specific facts, dates, numbers, and quotes from the documents. Identify any contradictions or gaps in the information provided.\n\nNotes:"
        return prompt

    def _create_answer_prompt(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Create a prompt for answer generation."""
        prompt = f"Question: {query}\n\nContext Documents:\n"
        for i, doc in enumerate(documents, 1):
            prompt += f"[Document {i}]: {doc['content'][:300]}...\n\n"  # Truncate for brevity
        prompt += f"Notes on the context:\n{notes}\n\nUsing the provided context documents and notes, please answer the question. Only include information that is supported by the context. If the context doesn't provide enough information, acknowledge the limitations in your answer.\n\nAnswer:"
        return prompt

    def generate_note(self, query: str, documents: List[Dict]) -> str:
        """Generate notes based on query and documents."""
        prompt = self._create_note_prompt(query, documents)

        if self.use_mistral:
            return self.mistral_chat.generate_response(prompt)
        elif self.model is None:  # For T5 and similar models
            result = self.generator(prompt, max_length=512, num_return_sequences=1, temperature=0.3)
            return result[0]["generated_text"].strip()
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_length=512, temperature=0.3, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def generate_answer(self, query: str, notes: str, documents: List[Dict]) -> str:
        """Generate final answer based on query, notes, and documents."""
        prompt = self._create_answer_prompt(query, notes, documents)

        if self.use_mistral:
            return self.mistral_chat.generate_response(prompt)
        elif self.model is None:  # For T5 and similar models
            result = self.generator(prompt, max_length=512, num_return_sequences=1, temperature=0.3)
            return result[0]["generated_text"].strip()
        else:  # For GPT and similar models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_length=512, temperature=0.3, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def generate_response(self, query: str, documents: List[Dict], return_notes: bool = False) -> Dict[str, Any]:
        """Generate a complete response using Chain-of-Note."""
        notes = self.generate_note(query, documents)
        answer = self.generate_answer(query, notes, documents)
        response = {"answer": answer}
        if return_notes:
            response["notes"] = notes
        return response
