"""
Command-line interface for Chain-of-Note RAG.
"""
import argparse
import os
import json
import sys
from typing import List, Optional

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from src.rag_system import ChainOfNoteRAG

def list_available_models():
    """List available models for embeddings and generation."""
    embedding_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/e5-large-v2",
        "intfloat/multilingual-e5-large"
    ]
    
    generation_models = [
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "tiiuae/falcon-7b-instruct"
    ]
    
    print("Available Embedding Models:")
    for model in embedding_models:
        print(f"  - {model}")
    
    print("\nAvailable Generation Models:")
    for model in generation_models:
        print(f"  - {model}")

def load_documents(input_path: str) -> DocumentLoader:
    """Load documents from various input sources.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        DocumentLoader with loaded documents
    """
    loader = DocumentLoader()
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
    
    # Check if it's a directory
    if os.path.isdir(input_path):
        print(f"Loading documents from directory: {input_path}")
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if not os.path.isfile(file_path):
                continue
                
            # Process based on file extension
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    loader.load_text(content, {"source": filename})
                    
            elif filename.endswith('.csv'):
                loader.load_csv(file_path, text_column="content")
                
            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if "content" in item:
                                metadata = {k: v for k, v in item.items() if k != "content"}
                                loader.load_text(item["content"], metadata)
                                
    # Single file
    elif input_path.endswith('.txt'):
        print(f"Loading text file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            loader.load_text(content, {"source": os.path.basename(input_path)})
            
    elif input_path.endswith('.csv'):
        print(f"Loading CSV file: {input_path}")
        loader.load_csv(input_path, text_column="content")
        
    elif input_path.endswith('.json'):
        print(f"Loading JSON file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "content" in item:
                        metadata = {k: v for k, v in item.items() if k != "content"}
                        loader.load_text(item["content"], metadata)
                        
    else:
        print(f"Unsupported file type: {input_path}")
        sys.exit(1)
    
    print(f"Loaded {len(loader.documents)} documents")
    return loader

def init_model(embedding_model_name: str, llm_model_name: str) -> ChainOfNoteRAG:
    """Initialize RAG model with specified embedding and LLM models.
    
    Args:
        embedding_model_name: Name of the embedding model
        llm_model_name: Name of the language model
        
    Returns:
        Initialized ChainOfNoteRAG instance
    """
    print(f"Initializing embedding model: {embedding_model_name}")
    
    if "sentence-transformers" in embedding_model_name:
        embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    else:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    print(f"Initializing language model: {llm_model_name}")
    rag_system = ChainOfNoteRAG(embedding_model=embedding_model, llm_model_name=llm_model_name)
    
    return rag_system

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Chain-of-Note RAG System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    index_parser.add_argument("--output", "-o", required=True, help="Output index file")
    index_parser.add_argument("--embedding-model", "-e", default="sentence-transformers/all-mpnet-base-v2", 
                              help="Embedding model to use")
    index_parser.add_argument("--chunk-size", "-c", type=int, default=500, 
                              help="Size of document chunks")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--index", "-i", required=True, help="Index file")
    query_parser.add_argument("--query", "-q", required=True, help="Query to process")
    query_parser.add_argument("--embedding-model", "-e", default="sentence-transformers/all-mpnet-base-v2", 
                             help="Embedding model to use")
    query_parser.add_argument("--llm-model", "-l", default="google/flan-t5-large", 
                             help="Language model to use")
    query_parser.add_argument("--top-k", "-k", type=int, default=5, 
                             help="Number of documents to retrieve")
    query_parser.add_argument("--show-notes", action="store_true", 
                             help="Show generated notes")
    query_parser.add_argument("--show-context", action="store_true", 
                             help="Show retrieved context")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--index", "-i", required=True, help="Index file")
    interactive_parser.add_argument("--embedding-model", "-e", default="sentence-transformers/all-mpnet-base-v2", 
                                   help="Embedding model to use")
    interactive_parser.add_argument("--llm-model", "-l", default="google/flan-t5-large", 
                                   help="Language model to use")
    interactive_parser.add_argument("--top-k", "-k", type=int, default=5, 
                                   help="Number of documents to retrieve")
    interactive_parser.add_argument("--show-notes", action="store_true", 
                                   help="Show generated notes")
    interactive_parser.add_argument("--show-context", action="store_true", 
                                   help="Show retrieved context")
                                   
    args = parser.parse_args()
    
    # Handle list-models command
    if args.command == "list-models":
        list_available_models()
        return
        
    # Handle index command
    elif args.command == "index":
        loader = load_documents(args.input)
        embedding_model = SentenceTransformerEmbeddings(model_name=args.embedding_model)
        document_store = embedding_model.document_store if hasattr(embedding_model, 'document_store') else None
        
        if document_store is None:
            from src.document_store import DocumentStore
            document_store = DocumentStore()
        
        # Process chunks and create embeddings
        chunks = loader.create_chunks(chunk_size=args.chunk_size)
        contents = [chunk["content"] for chunk in chunks]
        embeddings = embedding_model.embed_documents(contents)
        document_store.add_documents(chunks, embeddings)
        
        # Save the index
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        document_store.save(args.output)
        print(f"Index saved to {args.output}")
        
    # Handle query command
    elif args.command == "query":
        # Initialize the system
        rag_system = init_model(args.embedding_model, args.llm_model)
        
        # Load the index
        if not os.path.exists(args.index):
            print(f"Error: Index file '{args.index}' not found.")
            sys.exit(1)
            
        rag_system.document_store.load(args.index)
        print(f"Loaded index with {rag_system.document_store.index.ntotal} documents")
        
        # Process query
        response = rag_system.query(
            query=args.query,
            top_k=args.top_k,
            return_context=args.show_context,
            return_notes=args.show_notes
        )
        
        print("\n" + "=" * 60)
        print(f"Query: {args.query}")
        print("=" * 60)
        
        if args.show_notes and "notes" in response:
            print("\nGenerated Notes:")
            print("-" * 60)
            print(response["notes"])
            print("-" * 60)
            
        print("\nAnswer:")
        print("-" * 60)
        print(response["answer"])
        print("-" * 60)
        
        if args.show_context and "context" in response:
            print("\nRetrieved Context:")
            for i, doc in enumerate(response["context"], 1):
                print(f"\n[{i}] Score: {doc['score']:.4f}")
                print(f"Content: {doc['content'][:200]}...")
                if "metadata" in doc and doc["metadata"]:
                    print(f"Metadata: {doc['metadata']}")
                    
    # Handle interactive command
    elif args.command == "interactive":
        # Initialize the system
        rag_system = init_model(args.embedding_model, args.llm_model)
        
        # Load the index
        if not os.path.exists(args.index):
            print(f"Error: Index file '{args.index}' not found.")
            sys.exit(1)
            
        rag_system.document_store.load(args.index)
        print(f"Loaded index with {rag_system.document_store.index.ntotal} documents")
        
        print("\nEntering interactive mode. Type 'exit' to quit.")
        
        while True:
            try:
                query = input("\nEnter your query: ")
                
                if query.lower() in ("exit", "quit", "q"):
                    break
                    
                if not query.strip():
                    continue
                    
                # Process query
                response = rag_system.query(
                    query=query,
                    top_k=args.top_k,
                    return_context=args.show_context,
                    return_notes=args.show_notes
                )
                
                if args.show_notes and "notes" in response:
                    print("\nGenerated Notes:")
                    print("-" * 60)
                    print(response["notes"])
                    print("-" * 60)
                    
                print("\nAnswer:")
                print("-" * 60)
                print(response["answer"])
                print("-" * 60)
                
                if args.show_context and "context" in response:
                    print("\nRetrieved Context:")
                    for i, doc in enumerate(response["context"], 1):
                        print(f"\n[{i}] Score: {doc['score']:.4f}")
                        print(f"Content: {doc['content'][:200]}...")
                        if "metadata" in doc and doc["metadata"]:
                            print(f"Metadata: {doc['metadata']}")
                            
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
                
            except Exception as e:
                print(f"Error processing query: {e}")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
