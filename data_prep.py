"""
data_prep.py
------------
Downloads/prepares a high-quality corpus of technical documents (AI, ML, NLP, RAG)
and generates/formats a 15-question evaluation set with ground truth answers.

Output files:
- data/documents.json: Raw documents with metadata (title, text, source)
- data/eval_set.json: 15 (question, ground_truth, topic) evaluation pairs
"""

import json
import os
import requests
from config import DOCUMENTS_PATH, EVAL_SET_PATH, DATA_DIR


def fetch_wikipedia_summary(title: str) -> str:
    """Fetch text summary of a Wikipedia topic via REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    headers = {"User-Agent": "RAGEvalBot/1.0 (portfolio_eval_project)"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")
    except Exception as e:
        print(f"Warning: Could not fetch Wikipedia page for {title}: {e}")
    return ""


# High-quality fallback text corpus on AI/ML/RAG to ensure robust execution
DEFAULT_CORPUS = [
    {
        "id": "doc_1",
        "title": "Artificial Intelligence Overview",
        "topic": "AI Fundamentals",
        "text": (
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
            "that are programmed to think and learn like humans. Core subfields of AI include Machine Learning (ML), "
            "Computer Vision, Natural Language Processing (NLP), and Robotics. AI systems leverage algorithms, "
            "data, and computing power to perform tasks such as visual perception, speech recognition, decision-making, "
            "and language translation. Rule-based systems rely on explicit expert logic, whereas modern data-driven AI "
            "learns representations directly from large scale unstructured datasets using deep neural networks."
        )
    },
    {
        "id": "doc_2",
        "title": "Machine Learning and Neural Networks",
        "topic": "Machine Learning",
        "text": (
            "Machine Learning is a branch of AI focused on building applications that learn from data and improve "
            "their performance over time without being explicitly programmed. Supervised learning algorithms learn "
            "from labeled training datasets to map input features to output targets (e.g., classification and regression). "
            "Unsupervised learning discovers hidden patterns, clusters, or representations in unlabeled data (e.g., K-means, PCA). "
            "Reinforcement learning trains agents to make sequences of decisions by interacting with an environment to maximize cumulative reward. "
            "Deep Learning utilizes multi-layered artificial neural networks (ANNs) to capture non-linear patterns in high-dimensional data."
        )
    },
    {
        "id": "doc_3",
        "title": "Natural Language Processing and Transformers",
        "topic": "NLP",
        "text": (
            "Natural Language Processing (NLP) enables computers to analyze, interpret, and generate human language. "
            "Traditional NLP relied on n-grams, TF-IDF, and recurrent neural networks (RNNs/LSTMs). The introduction of the "
            "Transformer architecture in 2017 ('Attention Is All You Need' by Vaswani et al.) revolutionized NLP. Transformers replace "
            "sequential recurrent processing with a self-attention mechanism, allowing parallel processing across entire text sequences. "
            "Self-attention dynamically calculates context weights between all token pairs, allowing models like BERT and GPT to capture "
            "long-range dependencies far more effectively than traditional RNNs."
        )
    },
    {
        "id": "doc_4",
        "title": "Large Language Models and Generative AI",
        "topic": "LLMs",
        "text": (
            "Large Language Models (LLMs) are deep learning models trained on trillions of text tokens using self-supervised "
            "objectives (such as next-token prediction or masked language modeling). Prominent examples include OpenAI's GPT-4, "
            "Google's Gemini, and Meta's Llama. LLMs exhibit emergent capabilities, including multi-step reasoning, code generation, "
            "and in-context learning through prompt engineering. However, LLMs suffer from notable limitations, such as factual hallucination, "
            "knowledge staleness due to fixed training cutoffs, and lack of access to private enterprise data stores."
        )
    },
    {
        "id": "doc_5",
        "title": "Retrieval-Augmented Generation (RAG)",
        "topic": "RAG Architecture",
        "text": (
            "Retrieval-Augmented Generation (RAG) is an architectural framework designed to mitigate LLM hallucinations and knowledge cutoff "
            "limitations. RAG connects an generative LLM to an external knowledge vector database (e.g., Pinecone, Chroma). When a user submits "
            "a query, the RAG system first converts the query into a dense embedding, searches the vector store for the top-k most semantically "
            "similar document chunks (Retrieval phase), and appends these retrieved context chunks to the system prompt. The LLM then synthesizes "
            "a grounded, factual answer based strictly on the retrieved context (Generation phase). RAG ensures privacy, real-time factual updates, "
            "and verifiable source attribution without expensive model fine-tuning."
        )
    },
    {
        "id": "doc_6",
        "title": "Vector Databases and Dense Embeddings",
        "topic": "Vector Stores",
        "text": (
            "Dense vector embeddings represent unstructured text, images, or audio as numerical vectors in a continuous high-dimensional space. "
            "Models like sentence-transformers (all-MiniLM-L6-v2) or OpenAI text-embedding-3-small map semantically similar sentences to nearby "
            "vector coordinates. Vector databases like Pinecone are optimized for fast Approximate Nearest Neighbor (ANN) search across millions "
            "of vectors using similarity metrics such as Cosine Similarity, Dot Product, or Euclidean Distance. Pinecone serverless indexes "
            "support multi-tenant separation using namespaces, allowing isolated chunking and embedding experiments within a single database."
        )
    },
    {
        "id": "doc_7",
        "title": "RAG Evaluation and RAGAS Framework",
        "topic": "Evaluation Metrics",
        "text": (
            "Evaluating RAG pipelines requires assessing both retrieval quality and generation accuracy. RAGAS (Retrieval-Augmented Generation "
            "Assessment) is an open-source evaluation framework providing automated metrics without needing human annotators. Key RAGAS metrics include: "
            "1. Context Precision: Measures if retrieved context chunks relevant to the question are ranked near the top. "
            "2. Context Recall: Measures if all relevant ground truth information was retrieved in the context chunks. "
            "3. Faithfulness: Evaluates if the generated answer is strictly derived from the retrieved context (checking for hallucinations). "
            "4. Answer Relevancy: Measures how directly pertinent the generated response is to the user's initial prompt."
        )
    }
]

# 15 Evaluation Questions + Ground Truth Answers
DEFAULT_EVAL_SET = [
    {
        "id": "q1",
        "question": "What is Artificial Intelligence and what are its core subfields?",
        "ground_truth": "Artificial Intelligence is the simulation of human intelligence in machines. Its core subfields include Machine Learning, Computer Vision, Natural Language Processing, and Robotics.",
        "topic": "AI Fundamentals"
    },
    {
        "id": "q2",
        "question": "How does data-driven modern AI differ from rule-based AI systems?",
        "ground_truth": "Rule-based systems rely on explicit expert logic, whereas modern data-driven AI learns representations directly from large scale unstructured datasets using deep neural networks.",
        "topic": "AI Fundamentals"
    },
    {
        "id": "q3",
        "question": "What is Supervised Learning and how does it work?",
        "ground_truth": "Supervised learning is a branch of machine learning where algorithms learn from labeled training datasets to map input features to output targets for tasks like classification and regression.",
        "topic": "Machine Learning"
    },
    {
        "id": "q4",
        "question": "What is the difference between unsupervised learning and reinforcement learning?",
        "ground_truth": "Unsupervised learning discovers hidden patterns or clusters in unlabeled data, whereas reinforcement learning trains agents to make sequential decisions by interacting with an environment to maximize cumulative reward.",
        "topic": "Machine Learning"
    },
    {
        "id": "q5",
        "question": "What core innovation made Transformer models superior to traditional RNNs and LSTMs?",
        "ground_truth": "Transformers introduced self-attention mechanisms that enable parallel processing across text sequences, dynamically calculating context weights between token pairs to capture long-range dependencies better than sequential RNNs.",
        "topic": "NLP"
    },
    {
        "id": "q6",
        "question": "Who introduced the Transformer architecture and in which paper?",
        "ground_truth": "The Transformer architecture was introduced in 2017 by Vaswani et al. in the landmark paper 'Attention Is All You Need'.",
        "topic": "NLP"
    },
    {
        "id": "q7",
        "question": "What are Large Language Models (LLMs) and how are they trained?",
        "ground_truth": "LLMs are deep learning models trained on trillions of text tokens using self-supervised objectives such as next-token prediction or masked language modeling.",
        "topic": "LLMs"
    },
    {
        "id": "q8",
        "question": "What are three major limitations of standalone Large Language Models?",
        "ground_truth": "Three major limitations of standalone LLMs are factual hallucinations, knowledge staleness due to fixed training cutoffs, and lack of access to private enterprise data stores.",
        "topic": "LLMs"
    },
    {
        "id": "q9",
        "question": "What is Retrieval-Augmented Generation (RAG) and what problem does it solve?",
        "ground_truth": "RAG is an architectural framework that connects generative LLMs to external vector databases to mitigate factual hallucinations and knowledge cutoffs without model fine-tuning.",
        "topic": "RAG Architecture"
    },
    {
        "id": "q10",
        "question": "What are the two main phases of a RAG pipeline?",
        "ground_truth": "The two main phases of RAG are: 1) Retrieval phase (converting query into an embedding and fetching top-k relevant chunks from vector store) and 2) Generation phase (synthesizing a grounded answer using retrieved contexts).",
        "topic": "RAG Architecture"
    },
    {
        "id": "q11",
        "question": "What are dense vector embeddings and how are they used?",
        "ground_truth": "Dense vector embeddings represent unstructured text as numerical vectors in a continuous high-dimensional space where semantically similar text passages map to nearby vector coordinates.",
        "topic": "Vector Stores"
    },
    {
        "id": "q12",
        "question": "What similarity metrics do vector databases use for Approximate Nearest Neighbor search?",
        "ground_truth": "Vector databases use similarity metrics such as Cosine Similarity, Dot Product, and Euclidean Distance for Approximate Nearest Neighbor search.",
        "topic": "Vector Stores"
    },
    {
        "id": "q13",
        "question": "What is RAGAS and why is it useful for evaluating RAG systems?",
        "ground_truth": "RAGAS (Retrieval-Augmented Generation Assessment) is an open-source evaluation framework providing automated metrics to measure retrieval quality and generation accuracy without requiring human annotators.",
        "topic": "Evaluation Metrics"
    },
    {
        "id": "q14",
        "question": "What is the difference between Context Precision and Context Recall in RAGAS?",
        "ground_truth": "Context Precision measures if relevant context chunks are ranked near the top of retrieval results, whereas Context Recall measures if all relevant ground truth information was successfully retrieved in the contexts.",
        "topic": "Evaluation Metrics"
    },
    {
        "id": "q15",
        "question": "How does RAGAS measure Faithfulness in a RAG pipeline?",
        "ground_truth": "RAGAS measures Faithfulness by evaluating whether the generated answer is strictly derived from and supported by the retrieved context chunks, identifying any ungrounded hallucinations.",
        "topic": "Evaluation Metrics"
    }
]


def prepare_dataset():
    """Download/enrich corpus and save documents.json & eval_set.json."""
    print("Preparing technical corpus for RAG evaluation...")

    # Attempt to fetch live Wikipedia summaries to enrich corpus if possible
    wiki_topics = ["Artificial intelligence", "Machine learning", "Natural language processing", "Large language model"]
    documents = list(DEFAULT_CORPUS)
    
    for i, topic in enumerate(wiki_topics):
        summary = fetch_wikipedia_summary(topic)
        if summary and len(summary) > 100:
            documents.append({
                "id": f"wiki_{i+1}",
                "title": f"Wikipedia: {topic}",
                "topic": topic,
                "text": summary
            })

    # Save documents.json
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved {len(documents)} documents to {DOCUMENTS_PATH}")

    # Save eval_set.json
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_EVAL_SET, f, indent=2)
    print(f"Saved {len(DEFAULT_EVAL_SET)} eval Q&A pairs to {EVAL_SET_PATH}")


if __name__ == "__main__":
    prepare_dataset()
