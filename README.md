# RAG Evaluation Framework

A Python project to test how different chunking strategies and embedding models affect a Retrieval-Augmented Generation (RAG) system. It uses LangChain for the pipeline, Pinecone for vector storage, and RAGAS metrics to evaluate the results.

## What this project does

I wanted to compare different ways of splitting documents and generating embeddings to see which combination retrieves better context for a RAG application.

I tested four configurations using two chunking strategies and two embedding models.

| Configuration | Chunking | Embedding Model |
|---|---|---|
| `fixed_minilm` | Fixed-size (400 characters) | MiniLM-L6-v2 |
| `recursive_minilm` | Recursive (400 characters) | MiniLM-L6-v2 |
| `fixed_openai` | Fixed-size (400 characters) | OpenAI text-embedding-3-small |
| `recursive_openai` | Recursive (400 characters) | OpenAI text-embedding-3-small |

The evaluation uses a technical document collection covering AI, machine learning, NLP, LLMs, and RAG, along with a dataset of 15 questions and their reference answers.

## How it works

The project is split into a few Python scripts so each part can be run and tested separately.

1. **Data preparation (`data_prep.py`)** – Prepares the document collection and evaluation questions.
2. **Chunking (`chunking.py`)** – Splits documents using fixed-size and recursive character splitting.
3. **Embeddings and storage (`embed_and_upsert.py`)** – Generates embeddings and stores the document chunks in Pinecone.
4. **Retrieval and generation (`generate_answers.py`)** – Retrieves the top 3 chunks for each question and generates an answer using the available LLM configuration.
5. **Evaluation (`evaluate.py`)** – Calculates evaluation metrics and exports the results to CSV for comparison.

## Evaluation metrics

I used four metrics to understand different parts of the RAG pipeline:

- **Context Precision:** Checks whether the retrieved chunks are relevant and ranked well.
- **Context Recall:** Measures how much of the information needed to answer a question is present in the retrieved context.
- **Faithfulness:** Checks whether the generated answer is supported by the retrieved context.
- **Answer Relevancy:** Measures how well the generated answer addresses the original question.

These metrics help identify whether a problem comes from retrieval or from answer generation, rather than judging the system only by its final response.

## Results

The following are the benchmark results from the 15-question evaluation dataset.

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Overall Score |
|---|---:|---:|---:|---:|---:|
| `recursive_minilm` | 0.8889 | 0.9667 | 1.0000 | 0.5528 | 0.8521 |
| `fixed_minilm` | 0.8778 | 0.9000 | 1.0000 | 0.5416 | 0.8299 |
| `fixed_openai` | 0.1667 | 0.3067 | 1.0000 | 0.4044 | 0.4694 |
| `recursive_openai` | 0.0889 | 0.2044 | 1.0000 | 0.4656 | 0.4397 |

### What I found

The recursive chunking and MiniLM combination performed best in this evaluation, with an overall score of 0.8521.

It also achieved the highest context recall (0.9667), meaning the retrieved chunks covered most of the information needed to answer the evaluation questions.

MiniLM performed better than the OpenAI embedding model in these particular experiments. This doesn't necessarily mean it is better for every dataset; the results depend on the documents, queries, chunking configuration, and evaluation setup.

Faithfulness was 1.0 for all four configurations, while answer relevancy was lower. This is a useful reminder that an answer can be grounded in the retrieved context without fully addressing what the user asked.

## Tech stack

- Python
- LangChain
- Pinecone
- Hugging Face Sentence Transformers
- OpenAI Embeddings
- RAGAS
- Pandas

## Project structure

```text
.
├── config.py
├── data_prep.py
├── chunking.py
├── embed_and_upsert.py
├── generate_answers.py
├── evaluate.py
├── .env.example
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file based on `.env.example` and add the required API keys.

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
OPENAI_API_KEY=your_openai_api_key
```

Keep your `.env` file private and never commit API keys to GitHub.

### 4. Run the pipeline

Run the scripts in the following order:

```bash
python data_prep.py
python embed_and_upsert.py
python generate_answers.py
python evaluate.py
```

After evaluation, check the generated CSV files to compare the results across configurations.

## Possible improvements

- Increase the size and diversity of the evaluation dataset.
- Test different chunk sizes and overlap settings.
- Add regression tests to catch changes in retrieval quality.
- Evaluate retrieval and generation separately to identify failure points.
- Compare performance, latency, and API costs alongside evaluation scores.
