"""
chunking.py
-----------
Implements and exports two chunking strategies for document ingestion:
1. Fixed-size Chunking (CharacterTextSplitter)
2. Recursive Character Chunking (RecursiveCharacterTextSplitter)

Returns LangChain Document objects ready for embedding and vector storage.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class FixedSizeChunker:
    """Strategy 1: Fixed-size character chunking with overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )

    def split(self, documents: List[Dict[str, Any]]) -> List[Document]:
        langchain_docs = []
        for doc in documents:
            lc_doc = Document(
                page_content=doc["text"],
                metadata={
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "topic": doc.get("topic", "General"),
                    "strategy": "fixed",
                },
            )
            langchain_docs.append(lc_doc)
        
        return self.splitter.split_documents(langchain_docs)


class RecursiveChunker:
    """Strategy 2: Recursive character chunking respecting paragraph/sentence boundaries."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Dict[str, Any]]) -> List[Document]:
        langchain_docs = []
        for doc in documents:
            lc_doc = Document(
                page_content=doc["text"],
                metadata={
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "topic": doc.get("topic", "General"),
                    "strategy": "recursive",
                },
            )
            langchain_docs.append(lc_doc)

        return self.splitter.split_documents(langchain_docs)


def get_chunks(documents: List[Dict[str, Any]], strategy: str) -> List[Document]:
    """Helper function to split documents according to specified strategy ('fixed' or 'recursive')."""
    if strategy == "fixed":
        chunker = FixedSizeChunker()
    elif strategy == "recursive":
        chunker = RecursiveChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return chunker.split(documents)


if __name__ == "__main__":
    import json
    from config import DOCUMENTS_PATH

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    fixed_chunks = get_chunks(docs, "fixed")
    rec_chunks = get_chunks(docs, "recursive")

    print(f"Loaded {len(docs)} documents.")
    print(f"Fixed-size chunking produced: {len(fixed_chunks)} chunks.")
    print(f"Recursive chunking produced: {len(rec_chunks)} chunks.")

    if fixed_chunks and rec_chunks:
        print("\n--- Sample Fixed Chunk ---")
        print(fixed_chunks[0].page_content)
        print("\n--- Sample Recursive Chunk ---")
        print(rec_chunks[0].page_content)
