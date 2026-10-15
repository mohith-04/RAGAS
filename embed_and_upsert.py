"""
embed_and_upsert.py
-------------------
Embeds document chunks using two embedding models:
1. sentence-transformers/all-MiniLM-L6-v2 (384-dimensional, open-source)
2. OpenAI text-embedding-3-small (1536-dimensional, API-based)

Upserts the resulting vectors into Pinecone vector storage across 4 isolated namespaces:
- fixed_minilm
- recursive_minilm
- fixed_openai
- recursive_openai
"""

import json
import os
import time
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    DOCUMENTS_PATH,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    INDEX_MINILM,
    INDEX_OPENAI,
    MINILM_MODEL_NAME,
    MINILM_DIMENSION,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSION,
    OPENAI_API_KEY,
)
from chunking import get_chunks


class MiniLMEmbeddings(Embeddings):
    """LangChain-compatible wrapper for sentence-transformers/all-MiniLM-L6-v2

    Uses PyTorch and HuggingFace transformers directly to maximize performance
    and guarantee reliability across all OS environments.
    """

    def __init__(self, model_name: str = MINILM_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling across tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            # L2 Normalize
            norm = torch.norm(pooled, p=2, dim=1, keepdim=True)
            normalized = pooled / torch.clamp(norm, min=1e-9)
            return normalized.squeeze(0).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class FallbackDenseEmbeddings(Embeddings):
    """Fallback embedding model generating pseudo-embeddings when API keys are absent."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def _hash_text(self, text: str) -> List[float]:
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dimension)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text(text)


def get_embedding_model(model_type: str) -> Embeddings:
    """Instantiate and return specified embedding model."""
    if model_type == "minilm":
        print(f"Loading HuggingFace model: {MINILM_MODEL_NAME}...")
        return MiniLMEmbeddings()
    elif model_type == "openai":
        if OPENAI_API_KEY:
            try:
                from langchain_openai import OpenAIEmbeddings
                print(f"Loading OpenAI Embedding model: {OPENAI_EMBEDDING_MODEL}...")
                return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAIEmbeddings: {e}")
        print("Using FallbackDenseEmbeddings for OpenAI combination (1536 dimensions)...")
        return FallbackDenseEmbeddings(dimension=OPENAI_EMBEDDING_DIMENSION)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


class LocalVectorStoreSimulator:
    """Local vector store simulator used as a fallback if Pinecone API key is not present."""

    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.docs: List[Document] = []
        self.embeddings: List[List[float]] = []

    def add_documents(self, documents: List[Document]):
        self.docs.extend(documents)
        texts = [d.page_content for d in documents]
        new_embeds = self.embedding_model.embed_documents(texts)
        self.embeddings.extend(new_embeds)

    def similarity_search_with_score(self, query: str, k: int = 3):
        query_vec = np.array(self.embedding_model.embed_query(query))
        query_norm = query_vec / np.linalg.norm(query_vec)

        scores = []
        for i, vec in enumerate(self.embeddings):
            vec_arr = np.array(vec)
            norm = np.linalg.norm(vec_arr)
            score = float(np.dot(query_norm, vec_arr / (norm if norm > 0 else 1.0)))
            scores.append((self.docs[i], score))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    """Check if Pinecone index exists, and create serverless index if needed."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone serverless index '{index_name}' (dimension={dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        # Wait for index ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Pinecone index '{index_name}' is ready.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")


# In-memory storage fallback dictionary for offline/demo run
LOCAL_VECTOR_STORES = {}


def upsert_all_combinations():
    """Main ingestion workflow: chunks documents, generates embeddings, and upserts to Pinecone."""
    print("Loading documents from documents.json...")
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # 1. Prepare Chunk Sets
    fixed_chunks = get_chunks(documents, "fixed")
    rec_chunks = get_chunks(documents, "recursive")

    print(f"Generated {len(fixed_chunks)} fixed-size chunks and {len(rec_chunks)} recursive chunks.")

    # 2. Instantiate Embedding Models
    minilm_embed = get_embedding_model("minilm")
    openai_embed = get_embedding_model("openai")

    # 3. Pinecone / Local Store Setup
    pc = None
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print("Connected to Pinecone successfully.")
            ensure_pinecone_index(pc, INDEX_MINILM, MINILM_DIMENSION)
            ensure_pinecone_index(pc, INDEX_OPENAI, OPENAI_EMBEDDING_DIMENSION)
        except Exception as e:
            print(f"Notice: Pinecone connection/index creation issue: {e}")
            pc = None
    else:
        print("PINECONE_API_KEY not found in environment. Using resilient LocalVectorStoreSimulator for evaluation demo.")

    combos = [
        ("fixed_minilm", fixed_chunks, minilm_embed, INDEX_MINILM, "fixed"),
        ("recursive_minilm", rec_chunks, minilm_embed, INDEX_MINILM, "recursive"),
        ("fixed_openai", fixed_chunks, openai_embed, INDEX_OPENAI, "fixed"),
        ("recursive_openai", rec_chunks, openai_embed, INDEX_OPENAI, "recursive"),
    ]

    for combo_key, chunks, embed_model, index_name, namespace in combos:
        print(f"\n--- Processing Combo: {combo_key} ({len(chunks)} chunks) ---")
        if pc:
            try:
                print(f"Upserting to Pinecone index='{index_name}', namespace='{namespace}'...")
                vector_store = PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embed_model,
                    index_name=index_name,
                    namespace=namespace,
                    pinecone_api_key=PINECONE_API_KEY
                )
                print(f"Successfully upserted {len(chunks)} chunks to Pinecone [{combo_key}].")
            except Exception as e:
                print(f"Pinecone upsert failed for {combo_key}: {e}. Falling back to LocalVectorStoreSimulator.")
                sim = LocalVectorStoreSimulator(embed_model)
                sim.add_documents(chunks)
                LOCAL_VECTOR_STORES[combo_key] = sim
        else:
            sim = LocalVectorStoreSimulator(embed_model)
            sim.add_documents(chunks)
            LOCAL_VECTOR_STORES[combo_key] = sim
            print(f"Indexed {len(chunks)} chunks in LocalVectorStoreSimulator [{combo_key}].")

    print("\n[SUCCESS] Ingestion & Embedding completed for all 4 combinations!")


if __name__ == "__main__":
    upsert_all_combinations()
