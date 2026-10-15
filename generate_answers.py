"""
generate_answers.py
-------------------
Retrieves top-k context chunks from Pinecone (or vector store) for all 15 eval questions
across the 4 combinations, and generates LLM answers grounded in retrieved contexts.

Saves results to data/generated_results.json for evaluation by RAGAS.
"""

import json
import os
from typing import List, Dict, Any
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import (
    EVAL_SET_PATH,
    GENERATED_RESULTS_PATH,
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    NAMESPACES,
    TOP_K,
)
from embed_and_upsert import get_embedding_model, get_chunks, LocalVectorStoreSimulator
from chunking import get_chunks
from config import DOCUMENTS_PATH


def generate_llm_response(question: str, contexts: List[str]) -> str:
    """Generate answer grounded in retrieved contexts using OpenAI / LLM with fallback."""
    context_str = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    system_prompt = (
        "You are an AI assistant evaluating RAG pipeline performance. "
        "Answer the user's question directly and concisely, relying strictly on the retrieved context below.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}"
    )

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
            res = llm.invoke(system_prompt)
            return res.content.strip()
        except Exception as e:
            print(f"Notice: ChatOpenAI call failed ({e}). Using deterministic fallback synthesis.")

    # Rule-based synthesis for offline/demo evaluation
    if contexts:
        first_ctx = contexts[0]
        # Return first paragraph or sentence snippet
        snippet = first_ctx.split(". ")[0] + "." if ". " in first_ctx else first_ctx
        return f"Based on retrieved context: {snippet}"
    return "Insufficient context available to answer the question."


def get_retriever_for_combo(combo_key: str):
    """Return PineconeVectorStore retriever or LocalVectorStoreSimulator for specified combo."""
    namespace, index_name, model_type = NAMESPACES[combo_key]
    embed_model = get_embedding_model(model_type)

    if PINECONE_API_KEY:
        try:
            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embed_model,
                namespace=namespace,
                pinecone_api_key=PINECONE_API_KEY
            )
            return vector_store
        except Exception as e:
            print(f"Warning: Pinecone retriever initialization failed for {combo_key}: {e}")

    # Fallback local simulator
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    strategy = "fixed" if "fixed" in combo_key else "recursive"
    chunks = get_chunks(documents, strategy)
    sim = LocalVectorStoreSimulator(embed_model)
    sim.add_documents(chunks)
    return sim


def run_retrieval_and_generation() -> List[Dict[str, Any]]:
    """Execute retrieval + generation across all 15 questions for all 4 combinations."""
    print("Loading evaluation set from eval_set.json...")
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []

    for combo_key in NAMESPACES.keys():
        print(f"\n==========================================")
        print(f" Running Retrieval + Generation: [{combo_key}]")
        print(f"==========================================")
        
        retriever = get_retriever_for_combo(combo_key)

        for i, item in enumerate(eval_set):
            question = item["question"]
            ground_truth = item["ground_truth"]

            # Retrieve top-k chunks
            if isinstance(retriever, LocalVectorStoreSimulator):
                scored_docs = retriever.similarity_search_with_score(question, k=TOP_K)
                retrieved_chunks = [doc.page_content for doc, score in scored_docs]
            else:
                docs = retriever.similarity_search(question, k=TOP_K)
                retrieved_chunks = [doc.page_content for doc in docs]

            # Generate grounded answer
            generated_answer = generate_llm_response(question, retrieved_chunks)

            record = {
                "question_id": item["id"],
                "question": question,
                "ground_truth": ground_truth,
                "topic": item.get("topic", "General"),
                "combination": combo_key,
                "contexts": retrieved_chunks,
                "answer": generated_answer,
            }
            results.append(record)
            print(f"  [{i+1}/{len(eval_set)}] Q: {question[:45]}...")

    # Save to generated_results.json
    with open(GENERATED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Generated {len(results)} total evaluation records across 4 combinations.")
    print(f"Saved to {GENERATED_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_retrieval_and_generation()
