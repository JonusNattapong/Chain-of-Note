# API Reference

This document provides a high-level overview of the key classes and functions in the Chain-of-Note RAG system. For detailed information, refer to the source code in the `src/` directory.

## `src/rag_system.py`

### `class ChainOfNoteRAG`

This is the main class that orchestrates the entire RAG process.

*   **`__init__(self, embedding_model: Optional[EmbeddingModel] = None, llm_model_name: str = "google/flan-t5-large")`**: Initializes the `ChainOfNoteRAG` system.
    *   `embedding_model`:  An optional instance of an `EmbeddingModel` (defaults to `SentenceTransformerEmbeddings`).
    *   `llm_model_name`: The name of the language model to use for note and answer generation (defaults to "google/flan-t5-large").

*   **`add_documents(self, documents: List[Dict]) -> None`**: Adds a list of documents to the system. Each document should be a dictionary with at least a "content" key.

*   **`process_documents_from_loader(self, loader: DocumentLoader, chunk_size: int = 500) -> None`**: Processes documents from a `DocumentLoader` instance.

*   **`query(self, query: str, top_k: int = 5, return_context: bool = False, return_notes: bool = False) -> Dict[str, Any]`**: Processes a user query and returns a dictionary containing the answer, and optionally the generated notes and retrieved context documents.

## `src/chain_of_note.py`

### `class ChainOfNote`

This class handles the core Chain-of-Note logic: generating intermediate notes and synthesizing the final answer.

*   **`__init__(self, model_name: str = "google/flan-t5-large")`**: Initializes the `ChainOfNote` instance.
    *   `model_name`: The name of the language model to use.

*   **`generate_note(self, query: str, documents: List[Dict]) -> str`**: Generates intermediate notes based on the query and retrieved documents.

*   **`generate_answer(self, query: str, notes: str, documents: List[Dict]) -> str`**: Generates the final answer based on the query, notes, and retrieved documents.

*   **`generate_response(self, query: str, documents: List[Dict], return_notes: bool = False) -> Dict[str, Any]`**: Generates the complete response, including notes and the final answer.

## `src/embeddings.py`

### `class EmbeddingModel`

An abstract base class for embedding models.

*   **`embed_documents(self, documents: List[str]) -> np.ndarray`**: Embeds a list of document texts.
*   **`embed_query(self, query: str) -> np.ndarray`**: Embeds a user query.

### `class SentenceTransformerEmbeddings(EmbeddingModel)`

An implementation of `EmbeddingModel` using Sentence Transformers.

*    **`__init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2")`**: Initializes with a specific Sentence Transformer model.

### `class HuggingFaceEmbeddings(EmbeddingModel)`
An implementation using Hugging Face Transformers.

## `src/document_store.py`

### `class DocumentStore`

Manages the storage and retrieval of documents and their embeddings.

*   **`__init__(self, embedding_dim: int = 768)`**: Initializes the document store.
*   **`add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None`**: Adds documents and their embeddings to the store.
*   **`search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]`**: Searches for the top-k most similar documents to a query embedding.
*  **`save(self, file_path: str) -> None`**: Saves document store to a file
*  **`load(self, file_path: str) -> None`**: Loads document store from a file

## `src/data_loader.py`

### `class DocumentLoader`

Handles loading and preprocessing documents from various sources.

*   **`__init__(self)`**: Initializes the `DocumentLoader`.
*   **`load_text(self, text: str, metadata: Optional[Dict] = None) -> None`**: Loads a single text document.
*   **`load_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None`**: Loads multiple text documents.
*   **`load_csv(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]] = None) -> None`**: Loads documents from a CSV file.
*   **`create_chunks(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]`**: Splits the loaded documents into smaller chunks.
*   **`get_dataset(self) -> Dataset`**: Returns loaded documents

## `src/advanced_techniques.py`
### `class EnhancedChainOfNote(ChainOfNote)`
* **`__init__(self, model_name: str = "google/flan-t5-large", verification_model_name: str = "google/flan-t5-base")`**: Initializes EnhancedChainOfNote
* **`generate_response(self, query: str, documents: List[Dict], return_notes: bool = False, return_verification: bool = False) -> Dict[str, Any]`**: Generates a response to a query, with optional intermediate notes and claim verification.

## `src/evaluation.py`
### `class RAGEvaluator`
*   **`__init__(self)`**: Initializes RAGEvaluator
*   **`evaluate_response(self, query: str, generated_answer: str, reference_answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]`**: Evaluates a single generated response against a reference answer and retrieved documents.
*   **`compare_systems(self, queries: List[str], ground_truths: List[str], system_a_responses: List[Dict], system_b_responses: List[Dict]) -> Dict[str, Any]`**:  Compares the performance of two RAG systems (e.g., standard RAG vs. Chain-of-Note RAG) on a set of queries.
