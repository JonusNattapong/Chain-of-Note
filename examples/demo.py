"""
Demonstration of Chain-of-Note RAG system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote
from src.rag_system import ChainOfNoteRAG

from examples.sample_data import get_sample_documents, get_sample_queries

def run_demo():
    """Run a demonstration of the Chain-of-Note RAG system."""
    print("=== Chain-of-Note RAG Demonstration ===\n")
    
    # Step 1: Initialize the RAG system
    print("Initializing the RAG system...")
    embedding_model = SentenceTransformerEmbeddings()
    rag_system = ChainOfNoteRAG(embedding_model=embedding_model)
    
    # Step 2: Load sample documents
    print("Loading sample documents...")
    sample_documents = get_sample_documents()
    rag_system.add_documents(sample_documents)
    print(f"Added {len(sample_documents)} documents to the system\n")
    
    # Step 3: Process sample queries
    sample_queries = get_sample_queries()
    
    for i, query_info in enumerate(sample_queries, 1):
        query = query_info["query"]
        expected_topics = query_info["expected_topics"]
        
        print(f"Query {i}: {query}")
        print(f"Expected topics: {', '.join(expected_topics)}")
        
        # Process query with Chain-of-Note
        response = rag_system.query(query, top_k=3, return_context=True, return_notes=True)
        
        # Display results
        print("\nGenerated Notes:")
        print("-" * 40)
        print(response["notes"])
        print("-" * 40)
        
        print("\nFinal Answer:")
        print("-" * 40)
        print(response["answer"])
        print("-" * 40)
        
        print("\nRetrieved Documents:")
        for j, doc in enumerate(response["context"], 1):
            print(f"Document {j} (Score: {doc['score']:.4f}):")
            print(f"Topic: {doc['metadata'].get('topic', 'Unknown')}")
            print(f"{doc['content'][:200]}...")
            print()
        
        print("=" * 80 + "\n")

if __name__ == "__main__":
    run_demo()
