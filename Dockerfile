FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libfaiss-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for data and indexes
RUN mkdir -p data/indexes

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_CACHE_DIR=/app/model_cache
ENV DATA_DIR=/app/data

# Create model cache directory
RUN mkdir -p ${MODEL_CACHE_DIR}

# Expose API port
EXPOSE 5000

# Default command
CMD ["python", "main.py", "--interactive", "--index", "data/indexes/default_index.faiss"]
