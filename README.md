# Chain-of-Note RAG System

This project implements a Retrieval-Augmented Generation (RAG) system that uses the Chain-of-Note technique to reduce hallucinations in AI-generated responses.

## Overview

Chain-of-Note RAG is an enhanced approach to Retrieval-Augmented Generation that significantly reduces hallucinations by introducing an intermediate note-taking step between document retrieval and answer generation. The system first generates detailed notes from retrieved documents, then uses those notes to craft accurate responses.

### Benefits of Chain-of-Note over Traditional RAG:

1. **Reduced Hallucinations**: By explicitly capturing key facts from retrieved documents in intermediate notes, the model is less likely to generate incorrect information.

2. **Improved Attribution**: The note-taking step helps track the source of information, making the system's answers more transparent and verifiable.

3. **Enhanced Reasoning**: Breaking the process into retrieval → note-taking → answer generation creates a step-by-step reasoning chain that produces more accurate results.

## Features

- Document indexing and retrieval using sentence embeddings
- Chain-of-Note generation for improved reasoning
- Reduction of hallucinations through note-based context augmentation
- Integration with Hugging Face models
- Comprehensive evaluation metrics to compare performance with standard RAG

## Installation

```bash
pip install -r requirements.txt
```

For development installation:
```bash
pip install -e .
```

## Usage

### Quick Start

```python
from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.rag_system import ChainOfNoteRAG

# 1. Initialize the RAG system
embedding_model = SentenceTransformerEmbeddings()
rag_system = ChainOfNoteRAG(embedding_model=embedding_model)

# 2. Load documents
loader = DocumentLoader()
loader.load_text("Your document content here", {"source": "Example"})
rag_system.process_documents_from_loader(loader)

# 3. Query the system
response = rag_system.query(
    "Your question here", 
    top_k=5,
    return_context=True,
    return_notes=True
)

# 4. Access the results
notes = response["notes"]
answer = response["answer"]
context = response["context"]
```

### Examples

The project includes several examples:

1. **Basic Demo**: `examples/demo.py` - Demonstrates the core functionality with sample data
2. **RAG Comparison**: `examples/comparison.py` - Compares standard RAG with Chain-of-Note RAG
3. **Real-world Example**: `examples/real_world_example.py` - Uses Wikipedia data to showcase real-world application

## Architecture

The system consists of several key components:

1. **Document Loader**: Handles loading and chunking of documents
2. **Embedding Model**: Creates vector representations of documents and queries
3. **Document Store**: Vector database for efficient document retrieval
4. **Chain-of-Note**: Implements the note-taking and answer generation process
5. **RAG System**: Orchestrates the entire process from query to response

## Evaluation

The system includes an evaluation module that measures:

- Hallucination scores based on n-gram overlap
- Relevance of answers to queries
- ROUGE scores for answer quality
- Comparative metrics between standard RAG and Chain-of-Note RAG

## Citation

If you use this code in your research, please cite:

```
@software{chain_of_note_rag,
  author = {AI Developer},
  title = {Chain-of-Note RAG System},
  year = {2023},
  url = {https://github.com/example/chain-of-note-rag}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.