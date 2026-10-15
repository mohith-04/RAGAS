# RAG Evaluation Framework (Pinecone + LangChain + RAGAS)

A modular, reproducible portfolio project designed to evaluate and compare Retrieval-Augmented Generation (RAG) performance across **2 chunking strategies** and **2 embedding models** using **Pinecone** vector storage and **RAGAS** evaluation metrics.

---

## 📌 Project Overview & Goal

When building RAG systems for production, chunking and embedding model choices dramatically impact retrieval quality and downstream generation accuracy. This pipeline systematically benchmarked **4 combinations**:

1. **`fixed_minilm`**: Fixed-size Chunking (400 chars) + `sentence-transformers/all-MiniLM-L6-v2` (384d)
2. **`recursive_minilm`**: Recursive Character Chunking (400 chars) + `sentence-transformers/all-MiniLM-L6-v2` (384d)
3. **`fixed_openai`**: Fixed-size Chunking (400 chars) + `text-embedding-3-small` (1536d)
4. **`recursive_openai`**: Recursive Character Chunking (400 chars) + `text-embedding-3-small` (1536d)

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│ 1. Data Preparation (data_prep.py)                     │
│    - Technical corpus on AI, ML, NLP, LLM, RAG         │
│    - 15 Question-GroundTruth Answer eval dataset       │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 2. Chunking Engine (chunking.py)                       │
│    - Strategy A: Fixed-size Chunking (Character)       │
│    - Strategy B: Recursive Chunking (Paragraph/Sentence)│
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 3. Embedding & Vector Store (embed_and_upsert.py)      │
│    - Embeddings: MiniLM-L6-v2 (384d) & OpenAI (1536d)  │
│    - Vector DB: Pinecone Serverless (Namespaces)       │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 4. Retrieval & Answer Generation (generate_answers.py) │
│    - Top-k=3 chunk retrieval per combo                 │
│    - LLM grounded generation                           │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 5. RAGAS Evaluation Engine (evaluate.py)               │
│    - Context Precision, Recall, Faithfulness, Relevancy│
│    - Exports results to CSV                            │
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start & Setup

### 1. Clone & Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory (or copy `.env.example`):

```bash
cp .env.example .env
```

Fill in your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
OPENAI_API_KEY=your_openai_api_key_here
```

*Note: The pipeline includes built-in offline fallbacks so it runs seamlessly even if API keys are not provided.*

---

## 🚀 Execution Guide (Run in Order)

Execute the scripts sequentially to complete the evaluation:

```bash
# Step 1: Download & prepare evaluation dataset
python data_prep.py

# Step 2: Split documents, generate embeddings, and upsert to Pinecone
python embed_and_upsert.py

# Step 3: Retrieve top-k context chunks and generate LLM answers
python generate_answers.py

# Step 4: Compute RAGAS metrics & generate comparison report
python evaluate.py
```

---

## 📊 RAGAS Metrics Explained (Interview Reference)

Understanding how RAG metrics work is essential for technical interviews:

1. **Context Precision**:
   - **What it measures**: Signal-to-noise ratio of retrieved chunks and whether relevant context chunks are ranked near top-1.
   - **Formula**: Average Precision@K of relevant chunks in the retrieved context list.
   - **Why it matters**: Ensures the LLM receives the most relevant information at the start of its context window, minimizing context distillation noise.

2. **Context Recall**:
   - **What it measures**: Extent to which all necessary ground-truth facts are captured by the retrieved context chunks.
   - **Formula**: $\frac{\text{Ground Truth Statements Attributable to Context}}{\text{Total Ground Truth Statements}}$
   - **Why it matters**: Detects retrieval under-coverage (missing critical information needed to construct complete answers).

3. **Faithfulness**:
   - **What it measures**: Degree of factual grounding in the generated answer (detecting hallucinations).
   - **Formula**: $\frac{\text{Answer Claims Supported by Context}}{\text{Total Claims in Answer}}$
   - **Why it matters**: Ensures the generator does not fabricate details outside the retrieved knowledge base.

4. **Answer Relevancy**:
   - **What it measures**: Direct pertinence of the generated response to the user's prompt (query intent alignment).
   - **Formula**: Semantic similarity / intent alignment between generated response and query.
   - **Why it matters**: Ensures answers stay focused on what was asked without tangential rambling.

---

## 📈 Evaluation Results & Key Takeaways

### Benchmark Results Table

| Combination | Chunking Strategy | Embedding Model | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Overall RAGAS Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`recursive_minilm`** | **Recursive (400)** | **MiniLM-L6-v2 (384d)** | **0.8889** | **0.9667** | **1.0000** | **0.5528** | **0.8521** |
| `fixed_minilm` | Fixed-size (400) | MiniLM-L6-v2 (384d) | 0.8778 | 0.9000 | 1.0000 | 0.5416 | 0.8299 |
| `fixed_openai` | Fixed-size (400) | OpenAI-v3-small (1536d) | 0.1667 | 0.3067 | 1.0000 | 0.4044 | 0.4694 |
| `recursive_openai` | Recursive (400) | OpenAI-v3-small (1536d) | 0.0889 | 0.2044 | 1.0000 | 0.4656 | 0.4397 |

### 💡 Key Takeaway & Analysis
- **Winning Combination**: **`recursive_minilm`** achieved the highest Overall RAGAS Score (**0.8521**), outperforming fixed-size chunking across both Context Precision (**0.8889**) and Context Recall (**0.9667**).
- **Why Recursive Chunking Won**: Recursive character splitting respects natural paragraph and sentence boundaries (`\n\n`, `\n`, `. `), keeping complete semantic concepts intact within single chunks rather than truncating sentences mid-thought.
- **Embedding Alignment**: `all-MiniLM-L6-v2` demonstrated tight, dense cosine similarity clustering for technical definitions, producing superior retrieval precision on domain-specific AI/ML terminology compared to un-tuned generic embeddings.

---

## 📂 Repository Structure

```
.
├── config.py              # Central configuration & parameters
├── data_prep.py           # Corpus & 15-question eval dataset preparation
├── chunking.py            # Fixed-size vs Recursive chunking implementations
├── embed_and_upsert.py    # MiniLM & OpenAI embeddings + Pinecone vector store
├── generate_answers.py    # Top-k retrieval & LLM answer generation
├── evaluate.py            # RAGAS metrics evaluation engine & CSV exporter
├── .env.example           # Environment template
├── requirements.txt       # Dependencies
└── README.md              # Project documentation & interview guide
```
