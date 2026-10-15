"""
config.py
---------
Central configuration module for the RAG Evaluation Project.
Loads environment variables from .env and defines parameters for:
- Data paths
- Chunking strategies
- Embedding models
- Vector store (Pinecone) settings
- Evaluation settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Embedding Model Configurations
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSION = 384

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536

# Pinecone Settings
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

INDEX_MINILM = "rag-eval-384"
INDEX_OPENAI = "rag-eval-1536"

NAMESPACES = {
    "fixed_minilm": ("fixed", INDEX_MINILM, "minilm"),
    "recursive_minilm": ("recursive", INDEX_MINILM, "minilm"),
    "fixed_openai": ("fixed", INDEX_OPENAI, "openai"),
    "recursive_openai": ("recursive", INDEX_OPENAI, "openai"),
}

# Chunking Parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval & Evaluation
TOP_K = 3
EVAL_SET_PATH = DATA_DIR / "eval_set.json"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
GENERATED_RESULTS_PATH = DATA_DIR / "generated_results.json"
FINAL_RESULTS_CSV = RESULTS_DIR / "rag_evaluation_comparison.csv"
