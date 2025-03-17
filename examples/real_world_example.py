"""
Real-world example using Chain-of-Note RAG with Wikipedia data.
"""
import sys
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote
from src.rag_system import ChainOfNoteRAG

def fetch_wikipedia_content(topic):
    """Fetch content from Wikipedia for a given topic."""
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        content_div = soup.find(id="mw-content-text")
        if not content_div:
            return None
            
        # Get paragraphs
        paragraphs = content_div.find_all('p')
        content = "\n\n".join([p.get_text() for p in paragraphs])
        
        return {
            "content": content,
            "metadata": {
                "source": "Wikipedia",
                "topic": topic,
                "url": url
            }
        }
    except Exception as e:
        print(f"Error fetching {topic} from Wikipedia: {e}")
        return None

def run_real_world_example():
    """Run a real-world example with Wikipedia data."""
    print("=== Chain-of-Note RAG with Wikipedia Data ===\n")
    
    # Step 1: Initialize the RAG system
    print("Initializing the RAG system...")
    embedding_model = SentenceTransformerEmbeddings()
    rag_system = ChainOfNoteRAG(embedding_model=embedding_model)
    
    # Step 2: Fetch data from Wikipedia
    topics = [
        "Artificial intelligence",
        "Machine learning",
        "Natural language processing",
        "Retrieval-augmented generation",
        "Large language model",
        "Transformer (machine learning)",
        "ChatGPT",
        "GPT-4"
    ]
    
    print(f"Fetching data for {len(topics)} topics from Wikipedia...")
    documents = []
    
    for topic in tqdm(topics):
        doc = fetch_wikipedia_content(topic)
        if doc:
            documents.append(doc)
    
    print(f"Successfully fetched {len(documents)} documents")
    
    # Step 3: Load documents into the RAG system
    print("Loading documents into the RAG system...")
    
    # Create a document loader for chunking
    loader = DocumentLoader()
    for doc in documents:
        loader.load_text(doc["content"], doc["metadata"])
    
    # Process and add document chunks
    rag_system.process_documents_from_loader(loader, chunk_size=300)
    
    # Step 4: Example queries
    example_queries = [
        "What is the relationship between transformers and large language models?",
        "How does retrieval-augmented generation help reduce hallucinations?",
        "Compare and contrast GPT-4 with earlier versions.",
        "What are the main techniques used in modern NLP?",
        "Explain how machine learning is used in artificial intelligence."
    ]
    
    # Step 5: Process queries
    print("\nProcessing queries...\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"Query {i}: {query}")
        
        # Process query with Chain-of-Note
        response = rag_system.query(query, top_k=5, return_context=True, return_notes=True)
        
        # Display notes
        print("\nGenerated Notes:")
        print("-" * 60)
        print(response["notes"])
        print("-" * 60)
        
        # Display answer
        print("\nFinal Answer:")
        print("-" * 60)
        print(response["answer"])
        print("-" * 60)
        
        # Show sources
        print("\nSources:")
        sources = set()
        for doc in response["context"]:
            topic = doc["metadata"].get("topic", "Unknown")
            sources.add(topic)
        
        for source in sources:
            print(f"- {source}")
        
        print("\n" + "=" * 80 + "\n")
    
    # Save data for future use
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save Wikipedia content
    wiki_data = []
    for doc in documents:
        wiki_data.append({
            "topic": doc["metadata"]["topic"],
            "content": doc["content"],
            "url": doc["metadata"]["url"]
        })
    
    df = pd.DataFrame(wiki_data)
    df.to_csv(os.path.join(data_dir, "wikipedia_data.csv"), index=False)
    print(f"Saved Wikipedia data to {os.path.join(data_dir, 'wikipedia_data.csv')}")
    
    # Save RAG index
    index_path = os.path.join(data_dir, "rag_index.faiss")
    rag_system.document_store.save(index_path)
    print(f"Saved RAG index to {index_path}")

if __name__ == "__main__":
    run_real_world_example()
