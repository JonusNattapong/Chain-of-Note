# Chain-of-Note RAG Quickstart Guide

This guide provides step-by-step instructions to get started with the Chain-of-Note RAG system for reducing hallucinations in AI-generated responses.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/chain-of-note.git  
    cd chain-of-note
    ```
    Replace `https://github.com/your-username/chain-of-note.git` with the actual repository URL.

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
    
    Alternatively, you can install the packages manually:
    ```bash
    pip install transformers>=4.30.0 datasets>=2.12.0 sentence-transformers>=2.2.2 faiss-cpu>=1.7.4 torch>=2.0.0 numpy>=1.24.0 langchain>=0.0.267 pandas>=2.0.0 tqdm>=4.65.0 python-dotenv>=1.0.0
    ```

## Basic Usage

Here's a simple example to demonstrate the basic usage of the Chain-of-Note RAG system:

```python
from src.rag_system import ChainOfNoteRAG
from examples.sample_data import get_sample_documents, get_sample_queries

# Initialize the RAG system
rag_system = ChainOfNoteRAG()

# Load sample documents
sample_documents = get_sample_documents()
rag_system.add_documents(sample_documents)

# Get sample queries
sample_queries = get_sample_queries()

# Process a sample query
query = sample_queries[0]["query"]
response = rag_system.query(query, top_k=3, return_context=True, return_notes=True)

# Print the results
print(f"Query: {query}")
print(f"Answer: {response['answer']}")
print("Notes:")
print(response["notes"])
print("Context:")
for doc in response["context"]:
    print(f"- {doc['content'][:100]}...") # Print first 100 characters of each document

```

## System Workflow

![Workflow](public/image.png)

## Key Components

The Chain-of-Note RAG system consists of the following key components:

*   **`ChainOfNoteRAG`**: The main class that orchestrates the RAG process. It initializes the embedding model, document store, and Chain-of-Note components.
    *   `add_documents(documents)`: Adds documents to the system.
    *   `query(query, top_k=5, return_context=False, return_notes=False)`: Processes a query and returns the generated response, optionally including the retrieved context and generated notes.
*   **`EmbeddingModel`**: An interface for embedding models. The default implementation is `SentenceTransformerEmbeddings`.
*   **`DocumentStore`**: Stores document embeddings and provides methods for searching similar documents.
*   **`ChainOfNote`**: Generates the final response based on the retrieved documents and the user query, using the Chain-of-Note technique.

## Running the Demo

You can run a complete demonstration of the system using the provided `demo.py` script:

```bash
python examples/demo.py
```

This script will:

1.  Initialize the RAG system.
2.  Load sample documents.
3.  Process sample queries.
4.  Display the generated responses, retrieved documents, and intermediate notes.

## Advanced Usage

The `src/advanced_techniques.py` file contains examples of more advanced usage patterns.

## Further Reading

*   **Examples:** The `examples/` directory contains additional examples, including a Jupyter Notebook tutorial (`jupyter_tutorial.ipynb`), a real-world example (`real_world_example.py`), and a demo script (`demo.py`).
*   **API Reference:** For detailed API information, explore the source code directly, particularly the `src/` directory.
*   **Comparison:** The `examples/comparison.py` and `src/evaluation.py` files contain benchmarks against standard RAG.

## See Also
* [System Guide](system_guide.md)
* [API Reference](api_reference.md)