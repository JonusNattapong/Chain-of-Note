"""
Compare standard RAG with Chain-of-Note RAG.
"""
import sys
import os
from pprint import pprint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote
from src.rag_system import ChainOfNoteRAG
from src.evaluation import RAGEvaluator

from examples.sample_data import get_sample_documents, get_sample_queries

class StandardRAG:
    """Basic RAG implementation for comparison."""
    
    def __init__(self, embedding_model=None, model_name="google/flan-t5-large"):
        """Initialize standard RAG system."""
        self.embedding_model = embedding_model or SentenceTransformerEmbeddings()
        self.document_store = DocumentStore()
        
        # Use the same LLM as Chain-of-Note but without the note step
        self.chain_of_note = ChainOfNote(model_name=model_name)
    
    def add_documents(self, documents):
        """Add documents to the RAG system."""
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.embed_documents(contents)
        self.document_store.add_documents(documents, embeddings)
    
    def query(self, query, top_k=5, return_context=False):
        """Process a query through standard RAG (without notes)."""
        query_embedding = self.embedding_model.embed_query(query)
        retrieved_docs = self.document_store.search(query_embedding, top_k=top_k)
        
        # Create a standard RAG prompt without notes
        prompt = f"Question: {query}\n\nContext Information:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            prompt += f"[Document {i}]: {content}\n\n"
            
        prompt += "Based on the above context information, please answer the question concisely.\n\nAnswer:"
        
        # Generate answer directly without notes
        if hasattr(self.chain_of_note, 'generator'):
            result = self.chain_of_note.generator(prompt, max_length=512, num_return_sequences=1)
            answer = result[0]["generated_text"]
        else:
            inputs = self.chain_of_note.tokenizer(prompt, return_tensors="pt").to(self.chain_of_note.device)
            output = self.chain_of_note.model.generate(**inputs, max_length=512)
            answer = self.chain_of_note.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        
        response = {"answer": answer.strip()}
        
        if return_context:
            response["context"] = retrieved_docs
            
        return response

def run_comparison():
    """Compare standard RAG with Chain-of-Note RAG."""
    print("=== RAG System Comparison ===\n")
    
    # Initialize systems
    print("Initializing systems...")
    embedding_model = SentenceTransformerEmbeddings()
    
    standard_rag = StandardRAG(embedding_model=embedding_model)
    chain_of_note_rag = ChainOfNoteRAG(embedding_model=embedding_model)
    
    # Load documents
    print("Loading sample documents...")
    sample_documents = get_sample_documents()
    standard_rag.add_documents(sample_documents)
    chain_of_note_rag.add_documents(sample_documents)
    print(f"Added {len(sample_documents)} documents to both systems\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Process sample queries
    sample_queries = get_sample_queries()
    results = []
    
    for i, query_info in enumerate(sample_queries, 1):
        query = query_info["query"]
        expected_topics = query_info["expected_topics"]
        
        print(f"Query {i}: {query}")
        print(f"Expected topics: {', '.join(expected_topics)}")
        
        # Process with standard RAG
        standard_response = standard_rag.query(query, top_k=3, return_context=True)
        
        # Process with Chain-of-Note RAG
        chain_response = chain_of_note_rag.query(query, top_k=3, return_context=True, return_notes=True)
        
        # Prepare reference documents for evaluation
        reference_docs = [doc["content"] for doc in standard_response["context"]]
        
        # Compare results
        print("\nStandard RAG Answer:")
        print("-" * 40)
        print(standard_response["answer"])
        print("-" * 40)
        
        print("\nChain-of-Note RAG Notes:")
        print("-" * 40)
        print(chain_response["notes"])
        print("-" * 40)
        
        print("\nChain-of-Note RAG Answer:")
        print("-" * 40)
        print(chain_response["answer"])
        print("-" * 40)
        
        # Evaluate and compare
        comparison = evaluator.compare_systems(
            query=query,
            standard_rag_response=standard_response["answer"],
            chain_of_note_response=chain_response["answer"],
            reference_docs=reference_docs
        )
        
        print("\nEvaluation Metrics:")
        pprint(comparison)
        
        results.append({
            "query": query,
            "standard_rag": standard_response["answer"],
            "chain_of_note": chain_response["answer"],
            "notes": chain_response["notes"],
            "evaluation": comparison
        })
        
        print("=" * 80 + "\n")
    
    # Show summary
    print("=== Summary ===")
    
    hallucination_reductions = [r["evaluation"]["hallucination_reduction"] for r in results]
    avg_hallucination_reduction = sum(hallucination_reductions) / len(hallucination_reductions)
    
    print(f"Average hallucination reduction: {avg_hallucination_reduction:.4f}")
    print("A positive value indicates Chain-of-Note RAG reduces hallucinations.")
    
    if avg_hallucination_reduction > 0:
        print("✅ Chain-of-Note RAG successfully reduced hallucinations compared to standard RAG.")
    else:
        print("❌ Chain-of-Note RAG did not reduce hallucinations in this test.")

if __name__ == "__main__":
    run_comparison()
