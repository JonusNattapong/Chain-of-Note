"""
Web demo for Chain-of-Note RAG system using Streamlit.
"""
import os
import sys
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.document_store import DocumentStore
from src.chain_of_note import ChainOfNote
from src.rag_system import ChainOfNoteRAG
from src.advanced_techniques import EnhancedChainOfNote
from src.evaluation import RAGEvaluator

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

# Page configuration
st.set_page_config(
    page_title="Chain-of-Note RAG Demo",
    page_icon="📝",
    layout="wide"
)

# Helper functions
def format_source_references(context):
    """Format source references for display."""
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

def initialize_standard_rag():
    """Initialize standard RAG system for comparison."""
    # Create a basic RAG system without notes
    if st.session_state.rag_system is not None:
        embedding_model = st.session_state.rag_system.embedding_model
        from examples.comparison import StandardRAG
        st.session_state.standard_rag = StandardRAG(embedding_model=embedding_model)
        
        # Add the same documents
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

# Sidebar
st.sidebar.title("Chain-of-Note RAG")
st.sidebar.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/retrieval-augmented.png", width=280)
st.sidebar.markdown("---")

# Model configuration
st.sidebar.header("Model Configuration")

embedding_options = [
    "sentence-transformers/all-mpnet-base-v2", 
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]
embedding_model_name = st.sidebar.selectbox(
    "Embedding Model", 
    embedding_options,
    index=1  # Default to MiniLM for speed
)

llm_options = [
    "google/flan-t5-large",
    "google/flan-t5-base",
    "google/flan-t5-small"
]
llm_model_name = st.sidebar.selectbox(
    "Language Model", 
    llm_options,
    index=2  # Default to small for speed
)

# Advanced options
st.sidebar.header("Advanced Options")

use_enhanced = st.sidebar.checkbox("Use Enhanced Chain-of-Note", value=False)
top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)

if use_enhanced:
    use_self_critique = st.sidebar.checkbox("Use Self-Critique", value=True)
    use_source_ranking = st.sidebar.checkbox("Use Source Ranking", value=True)

# Initialize models
init_button = st.sidebar.button("Initialize Models")

# Main area
st.title("Chain-of-Note RAG Demo")

tab1, tab2, tab3, tab4 = st.tabs(["Query System", "Manage Documents", "Comparison", "History"])

with tab1:
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
            
        if st.button("Submit Query", key="query_btn", disabled=st.session_state.rag_system is None):
            if query.strip():
                with st.spinner("Processing query..."):
                    # Process with Chain-of-Note
                    chain_response = st.session_state.rag_system.query(
                        query=query,
                        top_k=top_k,
                        return_context=show_context,
                        return_notes=show_notes
                    )
                    
                    # Process with standard RAG if requested
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
                    
                    # Save to history
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
                        
                        # Source references
                        st.subheader("Source References")
                        refs = format_source_references(chain_response["context"])
                        for ref in refs:
                            st.markdown(f"- {ref}")

with tab2:
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
                
                # Add to RAG system if initialized
                if st.session_state.rag_system is not None:
                    st.session_state.rag_system.add_documents([new_doc])
                
                # Add to standard RAG if initialized
                if st.session_state.standard_rag is not None:
                    st.session_state.standard_rag.add_documents([new_doc])
                    
                st.success("Document added successfully!")
                
        st.markdown("---")
        st.subheader("Load Sample Documents")
        
        if st.button("Load Sample Documents"):
            from examples.sample_data import get_sample_documents
            sample_docs = get_sample_documents()
            
            # Add to state
            st.session_state.documents.extend(sample_docs)
            
            # Add to RAG system if initialized
            if st.session_state.rag_system is not None:
                st.session_state.rag_system.add_documents(sample_docs)
            
            # Add to standard RAG if initialized
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
            
        # Clear documents
        if st.button("Clear All Documents"):
            st.session_state.documents = []
            
            # Reinitialize RAG systems
            if st.session_state.rag_system is not None:
                embedding_model = st.session_state.rag_system.embedding_model
                if use_enhanced:
                    st.session_state.rag_system = ChainOfNoteRAG(
                        embedding_model=embedding_model,
                        llm_model_name=llm_model_name
                    )
                else:
                    st.session_state.rag_system = ChainOfNoteRAG(
                        embedding_model=embedding_model,
                        llm_model_name=llm_model_name
                    )
            
            if st.session_state.standard_rag is not None:
                st.session_state.standard_rag = None
                
            st.success("All documents cleared!")

with tab3:
    st.header("Compare Systems")
    
    if st.session_state.rag_system is None:
        st.warning("Please initialize the models first using the sidebar.")
    else:
        comparison_query = st.text_area("Enter a question for comparison:", height=100, key="comparison_query")
        
        if st.button("Run Comparison", key="compare_btn"):
            if comparison_query.strip():
                # Initialize standard RAG if needed
                if st.session_state.standard_rag is None:
                    initialize_standard_rag()
                
                with st.spinner("Running comparison..."):
                    # Process with both systems
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
                    
                    # Prepare for evaluation
                    reference_docs = [doc["content"] for doc in chain_response["context"]]
                    evaluator = RAGEvaluator()
                    
                    # Compare systems
                    comparison = evaluator.compare_systems(
                        query=comparison_query,
                        standard_rag_response=standard_response["answer"],
                        chain_of_note_response=chain_response["answer"],
                        reference_docs=reference_docs
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Standard RAG")
                        st.info(standard_response["answer"])
                        
                        metrics = {
                            "Hallucination Score": comparison["standard_rag"]["hallucination_score"],
                        }
                        
                        if "relevance" in comparison["standard_rag"]:
                            metrics["Relevance"] = comparison["standard_rag"]["relevance"]
                            
                        st.write("Metrics:")
                        for k, v in metrics.items():
                            st.write(f"- {k}: {v:.4f}")
                    
                    with col2:
                        st.subheader("Chain-of-Note RAG")
                        st.success(chain_response["answer"])
                        
                        metrics = {
                            "Hallucination Score": comparison["chain_of_note_rag"]["hallucination_score"],
                        }
                        
                        if "relevance" in comparison["chain_of_note_rag"]:
                            metrics["Relevance"] = comparison["chain_of_note_rag"]["relevance"]
                            
                        st.write("Metrics:")
                        for k, v in metrics.items():
                            st.write(f"- {k}: {v:.4f}")
                    
                    # Show improvement
                    hallucination_reduction = comparison["hallucination_reduction"]
                    st.subheader("Comparison Results")
                    if hallucination_reduction > 0:
                        st.success(f"✅ Chain-of-Note reduced hallucinations by {hallucination_reduction:.4f} points!")
                    else:
                        st.error(f"❌ Chain-of-Note did not reduce hallucinations. Difference: {hallucination_reduction:.4f}")
                    
                    # Show generated notes
                    with st.expander("View Generated Notes"):
                        st.write(chain_response["notes"])
                    
                    # Show comparison chart
                    metrics = {
                        "Hallucination Score": [
                            comparison["standard_rag"]["hallucination_score"],
                            comparison["chain_of_note_rag"]["hallucination_score"]
                        ]
                    }
                    
                    if "relevance" in comparison["standard_rag"] and "relevance" in comparison["chain_of_note_rag"]:
                        metrics["Relevance"] = [
                            comparison["standard_rag"]["relevance"],
                            comparison["chain_of_note_rag"]["relevance"]
                        ]
                    
                    df = pd.DataFrame(metrics, index=["Standard RAG", "Chain-of-Note RAG"])
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    df.plot(kind="bar", ax=ax)
                    ax.set_title("Comparison: Standard RAG vs Chain-of-Note RAG")
                    ax.set_ylabel("Score")
                    plt.xticks(rotation=0)
                    
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.3f')
                        
                    st.pyplot(fig)

with tab4:
    st.header("Query History")
    
    if not st.session_state.history:
        st.info("No queries have been processed yet.")
    else:
        # Export functionality
        if st.button("Export History as JSON"):
            history_json = export_history()
            if history_json:
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"chain_of_note_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Display history
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
                    with st.expander("View Generated Notes"):
                        st.write(entry["notes"])

# Initialize models when button is clicked
if init_button:
    with st.spinner("Initializing models..."):
        try:
            # Initialize embedding model
            embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
            
            # Initialize language model and RAG system
            if use_enhanced:
                from src.advanced_techniques import EnhancedChainOfNote
                chain_of_note = EnhancedChainOfNote(
                    model_name=llm_model_name,
                    use_self_critique=use_self_critique if 'use_self_critique' in locals() else True,
                    use_source_ranking=use_source_ranking if 'use_source_ranking' in locals() else True
                )
                st.session_state.rag_system = ChainOfNoteRAG(
                    embedding_model=embedding_model,
                    llm_model_name=llm_model_name
                )
                st.session_state.rag_system.chain_of_note = chain_of_note
            else:
                st.session_state.rag_system = ChainOfNoteRAG(
                    embedding_model=embedding_model,
                    llm_model_name=llm_model_name
                )
            
            # Add any existing documents
            if st.session_state.documents:
                st.session_state.rag_system.add_documents(st.session_state.documents)
            
            st.sidebar.success("Models initialized successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error initializing models: {str(e)}")
