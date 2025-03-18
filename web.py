"""
Web demo for Chain-of-Note RAG system using Streamlit.
"""
import os
import sys
import asyncio
from functools import wraps
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Configure environment before imports
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["STREAMLIT_IMPORT_BLACKLIST"] = "torch.classes"
os.environ["STREAMLIT_SERVER_RUNONFAVE"] = "false"

# Initialize asyncio event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def ensure_event_loop(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception as e:
                st.error(f"Failed to set event loop: {str(e)}")
                return None
        try:
            return f(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in async execution: {str(e)}")
            raise e
    return wrapper

# Import after environment is configured
import torch
from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings, MistralEmbeddings
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote, MistralAIChat
from src.rag_system import ChainOfNoteRAG
from src.advanced_techniques import EnhancedChainOfNote
from src.evaluation import RAGEvaluator

def format_source_references(context):
    """Format source references for display."""
    try:
        sources = {}
        for i, doc in enumerate(context, 1):
            topic = doc["metadata"].get("topic", "Unknown")
            source = doc["metadata"].get("source", "Unknown")
            key = f"{source}: {topic}"
            if key not in sources:
                sources[key] = []
            sources[key].append(i)

        refs = []
        for source, indices in sources.items():
            indices_str = ", ".join([f"[{i}]" for i in indices])
            refs.append(f"{source} {indices_str}")

        return refs
    except Exception as e:
        st.error(f"Error formatting source references: {str(e)}")
        return []

def initialize_standard_rag():
    """Initialize standard RAG system for comparison."""
    if st.session_state.rag_system is not None:
        embedding_model = st.session_state.rag_system.embedding_model
        from examples.comparison import StandardRAG
        st.session_state.standard_rag = StandardRAG(embedding_model=embedding_model)

        for doc in st.session_state.documents:
            st.session_state.standard_rag.add_documents([doc])
        return True
    return False

def save_history(query, standard_response=None, chain_response=None):
    """Save query and responses to history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "query": query,
    }

    if standard_response:
        entry["standard_answer"] = standard_response["answer"]

    if chain_response:
        entry["chain_answer"] = chain_response["answer"]
        if "notes" in chain_response:
            entry["notes"] = chain_response["notes"]

    st.session_state.history.append(entry)

def export_history():
    """Export history as JSON."""
    if not st.session_state.history:
        st.warning("No history to export.")
        return None

    history_json = json.dumps(st.session_state.history, indent=2)
    return history_json

@ensure_event_loop
def main():
    try:
        # Configure page
        st.set_page_config(
            page_title="Chain-of-Note RAG Demo",
            page_icon="📝",
            layout="wide"
        )

        # Initialize session state
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'standard_rag' not in st.session_state:
            st.session_state.standard_rag = None
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'index_path' not in st.session_state:
            st.session_state.index_path = None

        # Sidebar
        st.sidebar.title("Chain-of-Note RAG")
        st.sidebar.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/retrieval-augmented.png", width=280)
        st.sidebar.markdown("---")

        # Model configuration
        st.sidebar.header("Model Configuration")

        # API Configuration
        st.sidebar.subheader("API Configuration")
        api_provider = st.sidebar.selectbox(
            "Embedding Provider",
            ["HuggingFace", "Mistral AI"],
            index=0
        )

        if api_provider == "HuggingFace":
            embedding_options = [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ]
            embedding_model_name = st.sidebar.selectbox("Embedding Model", embedding_options, index=0)
            hf_token = st.sidebar.text_input("HuggingFace Token", type="password")
            if hf_token:
                os.environ["HUGGINGFACE_TOKEN"] = hf_token
        else:  # Mistral AI
            mistral_token = st.sidebar.text_input("Mistral API Key", type="password")
            if mistral_token:
                os.environ["MISTRAL_API_KEY"] = mistral_token
            embedding_model_name = "mistral-embed"  # Mistral's default embedding model

        # LLM Selection for Chain-of-Note
        st.sidebar.subheader("Language Model Configuration")
        use_mistral_llm = st.sidebar.checkbox("Use Mistral AI for Language Model", value=False)
        if use_mistral_llm:
            llm_model_name = st.sidebar.selectbox("Mistral AI Model", ["mistral-tiny", "mistral-small", "mistral-medium"], index=1)
        else:
            llm_options = ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small"]
            llm_model_name = st.sidebar.selectbox("Language Model", llm_options, index=0)


        # Advanced options
        st.sidebar.header("Advanced Options")
        use_enhanced = st.sidebar.checkbox("Use Enhanced Chain-of-Note", value=False)  # Placeholder
        top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)

        if use_enhanced:
            # Placeholder for future enhancements
            use_self_critique = st.sidebar.checkbox("Use Self-Critique", value=True)
            use_source_ranking = st.sidebar.checkbox("Use Source Ranking", value=True)

        # Initialize models button
        init_button = st.sidebar.button("Initialize Models")

        # Main area
        st.title("Chain-of-Note RAG Demo")
        tabs = st.tabs(["Query System", "Manage Documents", "Comparison", "History"])

        # Tab 0: Query System
        with tabs[0]:
            st.header("Query the RAG System")
            if st.session_state.rag_system is None:
                st.warning("Please initialize the models first using the sidebar.")
            else:
                query = st.text_area("Enter your question:", height=100)
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_notes = st.checkbox("Show Notes", value=True)
                with col2:
                    show_context = st.checkbox("Show Context", value=True)
                with col3:
                    compare_standard = st.checkbox("Compare with Standard RAG", value=False)

                if st.button("Submit Query", key="query_btn"):
                    if query.strip():
                        with st.spinner("Processing query..."):
                            chain_response = st.session_state.rag_system.query(
                                query=query,
                                top_k=top_k,
                                return_context=show_context,
                                return_notes=show_notes
                            )

                            # Handle standard RAG comparison if requested
                            standard_response = None
                            if compare_standard:
                                if st.session_state.standard_rag is None:
                                    if initialize_standard_rag():
                                         standard_response = st.session_state.standard_rag.query(
                                            query=query,
                                            top_k=top_k,
                                            return_context=show_context
                                        )
                                else:
                                    standard_response = st.session_state.standard_rag.query(
                                        query=query,
                                        top_k=top_k,
                                        return_context=show_context
                                    )

                            save_history(query, standard_response, chain_response)

                            # Display results
                            if show_notes and "notes" in chain_response:
                                st.subheader("Generated Notes")
                                st.info(chain_response["notes"])

                            st.subheader("Chain-of-Note Answer")
                            st.success(chain_response["answer"])

                            if compare_standard and standard_response:
                                st.subheader("Standard RAG Answer")
                                st.warning(standard_response["answer"])

                            if show_context and "context" in chain_response:
                                st.subheader("Retrieved Documents")
                                for i, doc in enumerate(chain_response["context"], 1):
                                    with st.expander(f"Document {i} (Score: {doc['score']:.4f})"):
                                        st.markdown(f"**Topic:** {doc['metadata'].get('topic', 'Unknown')}")
                                        st.markdown(f"**Source:** {doc['metadata'].get('source', 'Unknown')}")
                                        st.markdown(f"**Content:** {doc['content']}")
                                st.subheader("Source References")
                                refs = format_source_references(chain_response["context"])
                                for ref in refs:
                                    st.markdown(f"- {ref}")

        # Tab 1: Document Management
        with tabs[1]:
            st.header("Manage Documents")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Add Document")
                new_content = st.text_area("Document content:", height=150)
                new_source = st.text_input("Source:", "Example Source")
                new_topic = st.text_input("Topic:", "Example Topic")

                if st.button("Add Document"):
                    if new_content.strip():
                        new_doc = {
                            "content": new_content,
                            "metadata": {
                                "source": new_source,
                                "topic": new_topic
                            }
                        }
                        st.session_state.documents.append(new_doc)
                        if st.session_state.rag_system is not None:
                            st.session_state.rag_system.add_documents([new_doc])
                        if st.session_state.standard_rag is not None:
                            st.session_state.standard_rag.add_documents([new_doc])
                        st.success("Document added successfully!")

                st.markdown("---")
                st.subheader("Load Sample Documents")
                if st.button("Load Sample Documents"):
                    from examples.sample_data import get_sample_documents
                    sample_docs = get_sample_documents()
                    st.session_state.documents.extend(sample_docs)
                    if st.session_state.rag_system is not None:
                        st.session_state.rag_system.add_documents(sample_docs)
                    if st.session_state.standard_rag is not None:
                        st.session_state.standard_rag.add_documents(sample_docs)
                    st.success(f"Loaded {len(sample_docs)} sample documents!")

            with col2:
                st.subheader("Current Documents")
                if st.session_state.documents:
                    for i, doc in enumerate(st.session_state.documents):
                        with st.expander(f"Document {i+1} - {doc['metadata'].get('topic', 'Unknown')}"):
                            st.markdown(f"**Source:** {doc['metadata'].get('source', 'Unknown')}")
                            st.markdown(f"**Content:** {doc['content'][:200]}...")
                else:
                    st.info("No documents added yet.")

                if st.button("Clear All Documents"):
                    st.session_state.documents = []
                    st.session_state.rag_system = None
                    st.session_state.standard_rag = None
                    st.success("All documents cleared!")

        # Tab 2: Comparison
        with tabs[2]:
            st.header("System Comparison")
            if st.session_state.rag_system is not None:
                comparison_query = st.text_area("Enter comparison question:", height=100)
                if st.button("Run Comparison"):
                    if comparison_query.strip():
                        if st.session_state.standard_rag is None:
                            initialize_standard_rag()

                        with st.spinner("Running comparison..."):
                            chain_response = st.session_state.rag_system.query(
                                query=comparison_query,
                                top_k=top_k,
                                return_context=True,
                                return_notes=True
                            )

                            standard_response = st.session_state.standard_rag.query(
                                query=comparison_query,
                                top_k=top_k,
                                return_context=True
                            )

                            evaluator = RAGEvaluator()
                            reference_docs = [doc["content"] for doc in chain_response["context"]]

                            comparison = evaluator.compare_systems(
                                query=comparison_query,
                                standard_rag_response=standard_response["answer"],
                                chain_of_note_response=chain_response["answer"],
                                reference_docs=reference_docs
                            )

                            # Display comparison results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Standard RAG")
                                st.info(standard_response["answer"])
                                metrics = {
                                    "Hallucination Score": comparison["standard_rag"]["hallucination_score"],
                                }
                                st.write("Metrics:")
                                for k, v in metrics.items():
                                    st.write(f"- {k}: {v:.4f}")

                            with col2:
                                st.subheader("Chain-of-Note RAG")
                                st.success(chain_response["answer"])
                                metrics = {
                                    "Hallucination Score": comparison["chain_of_note_rag"]["hallucination_score"],
                                }
                                st.write("Metrics:")
                                for k, v in metrics.items():
                                    st.write(f"- {k}: {v:.4f}")

                            st.subheader("Comparison Results")
                            hallucination_reduction = comparison["hallucination_reduction"]
                            if hallucination_reduction > 0:
                                st.success(f"✅ Chain-of-Note reduced hallucinations by {hallucination_reduction:.4f} points!")
                            else:
                                st.error(f"❌ Chain-of-Note did not reduce hallucinations. Difference: {hallucination_reduction:.4f}")
            else:
                st.warning("Please initialize models first.")

        # Tab 3: History
        with tabs[3]:
            st.header("Query History")
            if not st.session_state.history:
                st.info("No queries have been processed yet.")
            else:
                if st.button("Export History"):
                    history_json = export_history()
                    if history_json:
                        st.download_button(
                            label="Download JSON",
                            data=history_json,
                            file_name=f"chain_of_note_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                for i, entry in enumerate(reversed(st.session_state.history)):
                    with st.expander(f"Query {len(st.session_state.history) - i}: {entry['timestamp']}"):
                        st.markdown(f"**Query:** {entry['query']}")
                        if "chain_answer" in entry:
                            st.markdown("**Chain-of-Note Answer:**")
                            st.success(entry["chain_answer"])
                        if "standard_answer" in entry:
                            st.markdown("**Standard RAG Answer:**")
                            st.info(entry["standard_answer"])
                        if "notes" in entry:
                            st.markdown("**Generated Notes:**")
                            st.info(entry["notes"])

        # Handle model initialization
        if init_button:
            with st.spinner("Initializing models..."):
                try:
                    # Initialize embedding model based on selected provider
                    use_mistral_embeddings = api_provider == "Mistral AI"
                    if use_mistral_embeddings:
                        embedding_model = MistralEmbeddings()
                    else:
                        embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

                    st.session_state.rag_system = ChainOfNoteRAG(
                        embedding_model_name=embedding_model_name,
                        use_mistral_embeddings=use_mistral_embeddings,
                        llm_model_name=llm_model_name,
                        use_mistral_llm=use_mistral_llm
                    )

                    if st.session_state.documents:
                        st.session_state.rag_system.add_documents(st.session_state.documents)

                    st.sidebar.success("Models initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error initializing models: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
