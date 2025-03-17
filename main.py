"""
Main entry point for the Chain-of-Note RAG system.
This file provides a complete implementation for production use.
"""
import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Optional, Any

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings, EmbeddingModel
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote
from src.advanced_techniques import EnhancedChainOfNote
from src.rag_system import ChainOfNoteRAG
from src.evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chain_of_note.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Chain-of-Note")

class ChainOfNoteSystem:
    """Complete Chain-of-Note RAG system for production use."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "google/flan-t5-large",
        use_enhanced: bool = False,
        use_self_critique: bool = True,
        use_source_ranking: bool = True,
        use_claim_verification: bool = False,
        temperature: float = 0.3,
        config_path: Optional[str] = None
    ):
        """Initialize the Chain-of-Note RAG system.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the language model
            use_enhanced: Whether to use enhanced Chain-of-Note
            use_self_critique: Whether to use self-critique (for enhanced mode)
            use_source_ranking: Whether to rank sources (for enhanced mode)
            use_claim_verification: Whether to verify claims (for enhanced mode)
            temperature: Temperature for generation
            config_path: Path to configuration file
        """
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.embedding_model_name = embedding_model_name
            self.llm_model_name = llm_model_name
            self.use_enhanced = use_enhanced
            self.use_self_critique = use_self_critique
            self.use_source_ranking = use_source_ranking
            self.use_claim_verification = use_claim_verification
            self.temperature = temperature
            
        self.rag_system = None
        self.document_loader = DocumentLoader()
        self.evaluator = RAGEvaluator()
        
        # Initialize the system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the RAG system with the specified models."""
        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        
        # Create embedding model
        if "sentence-transformers" in self.embedding_model_name:
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
        else:
            embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            
        # Create Chain-of-Note component
        if self.use_enhanced:
            logger.info("Using Enhanced Chain-of-Note")
            chain_of_note = EnhancedChainOfNote(
                model_name=self.llm_model_name,
                use_self_critique=self.use_self_critique,
                use_source_ranking=self.use_source_ranking,
                use_claim_verification=self.use_claim_verification,
                temperature=self.temperature
            )
        else:
            logger.info("Using Standard Chain-of-Note")
            chain_of_note = ChainOfNote(model_name=self.llm_model_name)
            
        # Create RAG system
        self.rag_system = ChainOfNoteRAG(embedding_model=embedding_model)
        self.rag_system.chain_of_note = chain_of_note
        
        logger.info("RAG system initialized successfully")
    
    def load_config(self, config_path: str):
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.embedding_model_name = config.get("embedding_model_name", "sentence-transformers/all-mpnet-base-v2")
            self.llm_model_name = config.get("llm_model_name", "google/flan-t5-large")
            self.use_enhanced = config.get("use_enhanced", False)
            self.use_self_critique = config.get("use_self_critique", True)
            self.use_source_ranking = config.get("use_source_ranking", True)
            self.use_claim_verification = config.get("use_claim_verification", False)
            self.temperature = config.get("temperature", 0.3)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_config(self, config_path: str):
        """Save current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration
        """
        config = {
            "embedding_model_name": self.embedding_model_name,
            "llm_model_name": self.llm_model_name,
            "use_enhanced": self.use_enhanced,
            "use_self_critique": self.use_self_critique,
            "use_source_ranking": self.use_source_ranking,
            "use_claim_verification": self.use_claim_verification,
            "temperature": self.temperature
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_documents(self, file_path: str, chunk_size: int = 500):
        """Load documents from file or directory.
        
        Args:
            file_path: Path to file or directory
            chunk_size: Size of document chunks
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Path does not exist: {file_path}")
                return False
                
            if os.path.isdir(file_path):
                # Load from directory
                logger.info(f"Loading documents from directory: {file_path}")
                for filename in os.listdir(file_path):
                    self._load_single_file(os.path.join(file_path, filename))
            else:
                # Load single file
                self._load_single_file(file_path)
                
            # Process documents
            chunks = self.document_loader.create_chunks(chunk_size=chunk_size)
            self.rag_system.add_documents(chunks)
            
            logger.info(f"Loaded and processed {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _load_single_file(self, file_path: str):
        """Load a single file based on its extension.
        
        Args:
            file_path: Path to the file
        """
        if not os.path.isfile(file_path):
            return
            
        try:
            # Handle different file types
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.document_loader.load_text(content, {"source": os.path.basename(file_path)})
                    
            elif file_path.endswith('.csv'):
                self.document_loader.load_csv(file_path, text_column="content")
                
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if "content" in item:
                                metadata = {k: v for k, v in item.items() if k != "content"}
                                self.document_loader.load_text(item["content"], metadata)
            
            logger.info(f"Loaded file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
    
    def save_index(self, index_path: str):
        """Save the current vector index.
        
        Args:
            index_path: Path to save the index
        """
        try:
            directory = os.path.dirname(os.path.abspath(index_path))
            os.makedirs(directory, exist_ok=True)
            
            self.rag_system.document_store.save(index_path)
            logger.info(f"Index saved to {index_path}")
            
            # Save document metadata separately
            metadata_path = os.path.join(directory, "document_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.rag_system.document_store.documents, f, indent=2)
                
            logger.info(f"Document metadata saved to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_path: str):
        """Load a previously saved vector index.
        
        Args:
            index_path: Path to the index file
        """
        try:
            if not os.path.exists(index_path):
                logger.error(f"Index file not found: {index_path}")
                return False
                
            self.rag_system.document_store.load(index_path)
            
            # Try to load document metadata
            directory = os.path.dirname(os.path.abspath(index_path))
            metadata_path = os.path.join(directory, "document_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.rag_system.document_store.documents = json.load(f)
            
            logger.info(f"Loaded index with {self.rag_system.document_store.index.ntotal} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        return_context: bool = False,
        return_notes: bool = False,
        return_verification: bool = False
    ) -> Dict[str, Any]:
        """Process a query through the RAG system.
        
        Args:
            query_text: User query
            top_k: Number of documents to retrieve
            return_context: Whether to include retrieved documents
            return_notes: Whether to include generated notes
            return_verification: Whether to include verification results
            
        Returns:
            Dictionary with query response
        """
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Check if we have an index
            if self.rag_system.document_store.index.ntotal == 0:
                logger.warning("No documents in the index")
                return {"error": "No documents have been loaded", "answer": "I don't have any information to provide. Please load some documents first."}
            
            # Process query with enhanced options if available
            if isinstance(self.rag_system.chain_of_note, EnhancedChainOfNote) and return_verification:
                # Need to use the enhanced method directly
                query_embedding = self.rag_system.embedding_model.embed_query(query_text)
                retrieved_docs = self.rag_system.document_store.search(query_embedding, top_k=top_k)
                
                response = self.rag_system.chain_of_note.generate_response(
                    query=query_text,
                    documents=retrieved_docs,
                    return_notes=return_notes,
                    return_verification=return_verification
                )
                
                if return_context:
                    response["context"] = retrieved_docs
            else:
                # Use the standard query method
                response = self.rag_system.query(
                    query=query_text,
                    top_k=top_k,
                    return_context=return_context,
                    return_notes=return_notes
                )
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": str(e), "answer": "Sorry, an error occurred while processing your query."}
    
    def evaluate(self, query: str, reference_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate system performance on a query.
        
        Args:
            query: Query to evaluate
            reference_answer: Optional reference answer
            
        Returns:
            Evaluation metrics
        """
        try:
            # Process with Chain-of-Note RAG
            chain_response = self.query(query, top_k=5, return_context=True)
            
            # Extract reference docs
            if "context" in chain_response:
                reference_docs = [doc["content"] for doc in chain_response["context"]]
            else:
                logger.warning("No context available for evaluation")
                return {"error": "No context available for evaluation"}
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_response(
                query=query,
                response=chain_response["answer"],
                reference_docs=reference_docs,
                reference_answer=reference_answer
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e)}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Chain-of-Note RAG System")
    
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--embedding-model", "-e", help="Name of the embedding model")
    parser.add_argument("--llm-model", "-l", help="Name of the language model")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced Chain-of-Note")
    parser.add_argument("--index", "-i", help="Path to saved index")
    parser.add_argument("--documents", "-d", help="Path to documents")
    parser.add_argument("--save-index", "-s", help="Path to save index")
    parser.add_argument("--query", "-q", help="Query to process")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize system
    system_kwargs = {}
    if args.config:
        system_kwargs["config_path"] = args.config
    if args.embedding_model:
        system_kwargs["embedding_model_name"] = args.embedding_model
    if args.llm_model:
        system_kwargs["llm_model_name"] = args.llm_model
    if args.enhanced:
        system_kwargs["use_enhanced"] = True
    
    system = ChainOfNoteSystem(**system_kwargs)
    
    # Load index if specified
    if args.index:
        if not system.load_index(args.index):
            sys.exit(1)
    
    # Load documents if specified
    if args.documents:
        if not system.load_documents(args.documents):
            sys.exit(1)
            
    # Save index if specified
    if args.save_index:
        system.save_index(args.save_index)
    
    # Process query if specified
    if args.query:
        response = system.query(
            args.query,
            top_k=args.top_k,
            return_context=True,
            return_notes=True
        )
        
        print("\n" + "=" * 60)
        print(f"Query: {args.query}")
        print("=" * 60)
        
        if "notes" in response:
            print("\nGenerated Notes:")
            print("-" * 60)
            print(response["notes"])
            print("-" * 60)
            
        print("\nAnswer:")
        print("-" * 60)
        print(response["answer"])
        print("-" * 60)
        
        if "context" in response:
            print("\nRetrieved Context:")
            for i, doc in enumerate(response["context"], 1):
                print(f"\n[{i}] Score: {doc['score']:.4f}")
                print(f"Content: {doc['content'][:200]}...")
                if "metadata" in doc and doc["metadata"]:
                    print(f"Metadata: {doc['metadata']}")
    
    # Run in interactive mode if specified
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        
        while True:
            try:
                query = input("\nEnter your query: ")
                
                if query.lower() in ("exit", "quit", "q"):
                    break
                    
                if not query.strip():
                    continue
                    
                # Process query
                response = system.query(
                    query_text=query,
                    top_k=args.top_k,
                    return_context=True,
                    return_notes=True
                )
                
                if "notes" in response:
                    print("\nGenerated Notes:")
                    print("-" * 60)
                    print(response["notes"])
                    print("-" * 60)
                    
                print("\nAnswer:")
                print("-" * 60)
                print(response["answer"])
                print("-" * 60)
                
                if "context" in response:
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
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
