# Chain-of-Note RAG System Guide

This document provides a detailed overview of the Chain-of-Note RAG system's architecture, components, and workflow.

## System Architecture

The Chain-of-Note RAG system is designed to improve the accuracy and reduce hallucinations in AI-generated responses by incorporating a note-taking step. The system consists of the following key components:

1.  **Document Loader:** Responsible for loading documents from various sources and chunking them into manageable pieces for processing.
2.  **Embedding Model:** Generates vector representations (embeddings) of documents and user queries. These embeddings capture the semantic meaning of the text.
3.  **Document Store:** A vector database that stores document embeddings and provides efficient similarity search capabilities to retrieve relevant documents for a given query.
4.  **Chain-of-Note:** This component implements the core logic of the system. It takes the user query and retrieved documents as input, generates intermediate notes based on the documents, and then uses these notes to construct the final answer.
5.  **RAG System:** The main class (`ChainOfNoteRAG`) that orchestrates the entire process, from receiving a user query to generating the final response.

## Workflow

The system operates in the following steps:

1.  **Document Ingestion:**
    *   Documents are loaded using the `DocumentLoader`.
    *   The `DocumentLoader` splits the documents into smaller chunks.
    *   The `EmbeddingModel` generates embeddings for each chunk.
    *   The chunks and their embeddings are stored in the `DocumentStore`.

2.  **Query Processing:**
    *   The user submits a query.
    *   The `EmbeddingModel` generates an embedding for the query.
    *   The `DocumentStore` searches for documents with embeddings similar to the query embedding.
    *   The top-k most relevant documents are retrieved.
    *   The `ChainOfNote` component generates notes based on the retrieved documents and the query.
    *   The `ChainOfNote` component generates the final answer based on the generated notes.
    *   The system returns the answer, and optionally the generated notes and retrieved documents (context).

## Workflow Diagram

https://www.mermaidchart.com/raw/7c1afbd8-24c8-4681-9a80-a877bebd1b63?theme=light&version=v0.1&format=svg

## Components in Detail

*   **`src/rag_system.py`:** Contains the `ChainOfNoteRAG` class, which is the main entry point for the system.
*   **`src/chain_of_note.py`:** Contains the `ChainOfNote` class, responsible for note generation and answer synthesis.
*   **`src/embeddings.py`:** Contains the `EmbeddingModel` interface and implementations (e.g., `SentenceTransformerEmbeddings`).
*   **`src/document_store.py`:** Contains the `DocumentStore` class, which manages document storage and retrieval.
*   **`src/data_loader.py`:** Contains the `DocumentLoader` class, responsible for loading and preprocessing documents.
