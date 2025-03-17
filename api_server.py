"""
API server for Chain-of-Note RAG System.
Provides RESTful endpoints for document loading, querying, and evaluation.
"""
import os
import json
import logging
import argparse
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename

from main import ChainOfNoteSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Chain-of-Note-API")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['INDEX_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indexes')

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

# Global system instance
system = None

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    if system is None:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    return jsonify({
        "status": "ok", 
        "document_count": system.rag_system.document_store.index.ntotal if system.rag_system else 0
    })

@app.route("/api/query", methods=["POST"])
def process_query():
    """Process a query and return results."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
    
    # Get request data
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data["query"]
    
    # Optional parameters
    top_k = data.get("top_k", 5)
    return_context = data.get("return_context", False)
    return_notes = data.get("return_notes", True)
    return_verification = data.get("return_verification", False)
    
    # Process query
    try:
        response = system.query(
            query_text=query,
            top_k=top_k,
            return_context=return_context,
            return_notes=return_notes,
            return_verification=return_verification
        )
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_documents():
    """Upload documents to the system."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploads and process documents
    uploaded_files = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
    
    # Process uploaded files
    try:
        for file_path in uploaded_files:
            system.load_documents(file_path)
        
        return jsonify({
            "message": f"Successfully processed {len(uploaded_files)} files",
            "document_count": system.rag_system.document_store.index.ntotal
        })
    except Exception as e:
        logger.error(f"Error processing uploads: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/index", methods=["GET"])
def get_indexes():
    """Get list of available indexes."""
    try:
        indexes = []
        for filename in os.listdir(app.config['INDEX_FOLDER']):
            if filename.endswith('.faiss'):
                indexes.append(filename)
        
        return jsonify({"indexes": indexes})
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/index/<name>", methods=["POST"])
def save_index(name):
    """Save the current index."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        # Secure the filename
        safe_name = secure_filename(name)
        if not safe_name.endswith('.faiss'):
            safe_name += '.faiss'
        
        index_path = os.path.join(app.config['INDEX_FOLDER'], safe_name)
        system.save_index(index_path)
        
        return jsonify({
            "message": f"Index saved as {safe_name}",
            "path": index_path
        })
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/index/<name>", methods=["GET"])
def load_index(name):
    """Load an index."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        # Secure the filename
        safe_name = secure_filename(name)
        if not safe_name.endswith('.faiss'):
            safe_name += '.faiss'
        
        index_path = os.path.join(app.config['INDEX_FOLDER'], safe_name)
        
        if not os.path.exists(index_path):
            return jsonify({"error": f"Index file not found: {safe_name}"}), 404
            
        result = system.load_index(index_path)
        
        if result:
            return jsonify({
                "message": f"Successfully loaded index {safe_name}",
                "document_count": system.rag_system.document_store.index.ntotal
            })
        else:
            return jsonify({"error": "Failed to load index"}), 500
            
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def evaluate_query():
    """Evaluate system performance on a query."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
    
    # Get request data
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data["query"]
    reference_answer = data.get("reference_answer")
    
    try:
        metrics = system.evaluate(query, reference_answer)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error evaluating query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/count", methods=["GET"])
def document_count():
    """Get the number of documents in the system."""
    if system is None:
        return jsonify({"error": "System not initialized"}), 500
        
    try:
        count = system.rag_system.document_store.index.ntotal
        return jsonify({"document_count": count})
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return jsonify({"error": str(e)}), 500

def initialize_system(config_path=None):
    """Initialize the Chain-of-Note system."""
    global system
    
    try:
        if config_path and os.path.exists(config_path):
            logger.info(f"Initializing system with configuration: {config_path}")
            system = ChainOfNoteSystem(config_path=config_path)
        else:
            logger.info("Initializing system with default settings")
            system = ChainOfNoteSystem()
            
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chain-of-Note RAG API Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--index", help="Path to index file to load on startup")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize system
    if not initialize_system(args.config):
        logger.error("Failed to initialize the system. Exiting.")
        exit(1)
    
    # Load index if specified
    if args.index and os.path.exists(args.index):
        logger.info(f"Loading index: {args.index}")
        if not system.load_index(args.index):
            logger.error("Failed to load the specified index. Continuing without index.")
    
    # Start the server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
