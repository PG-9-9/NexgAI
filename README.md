# MedBot: A Medical Q\&A Chatbot

Below is my submission for NexGAI challenge. Currently titled (MedBot) is a RAG system tailored to answer multiple-choice medical questions using **only** the MedMCQA dataset. By indexing every question, option, and explanation, MedBot guarantees that its responses remain grounded in verified medical knowledge without resorting to general web data or hallucinations.( I hope I covered most important topics in challenge :) ).

PS: To run the app manually or with Docker, you need to set up the environment keys—use a .env file (stored in root folder) for local runs or API keys if you’re running through Docker. If that sounds confusing, just follow the README :)

---

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Architecture & Component Breakdown](#architecture--component-breakdown)

   * [1. Data Indexing (`store_index.py`)](#1-data-indexing-store_indexpy)
   * [2. Embeddings Strategy](#2-embeddings-strategy)
   * [3. LangGraph Pipeline](#3-langgraph-pipeline)
   * [4. Fallback & Reranking](#4-fallback--reranking)
   * [5. FastAPI Service & Concurrency](#5-fastapi-service--concurrency)
   * [6. Observability & Monitoring](#6-observability--monitoring)
   * [7. Caching & Rate Limiting](#7-caching--rate-limiting)
3. [Installation & Setup](#installation--setup)

   * [Requirements & Environment Variables](#requirements--environment-variables)
   * [Manual Run (Local)](#manual-run-local)
   * [Docker Build & Run](#docker-build--run)
4. [Available Endpoints & UI](#available-endpoints--ui)
5. [Reproducing the Indexing Pipeline](#reproducing-the-indexing-pipeline)
6. [Appendix: Configuration Options](#appendix-configuration-options)

---

## Overview & Motivation

MedBot addresses a common problem in open-domain chatbots: hallucination. By limiting its knowledge base to the MedMCQA dataset, MedBot ensures that every answer is traceable back to a specific question, set of options, and its official explanation. This makes the system both reliable and transparent—critical in a medical context.

---

## Architecture & Component Breakdown

### 1. Data Indexing (`store_index.py`)

* **Purpose**: Convert the MedMCQA dataset into a vector index for fast retrieval.

* **Key Steps**:

  1. **Argument Parsing**: `--data_percent` (0.0–1.0) allows sampling a subset of examples for testing.
  2. **Dataset Load**: Uses `datasets.load_dataset('openlifescienceai/medmcqa', split='train')`.
  3. **Text & Metadata Extraction**:

     * Formats each example as:

       ```
       Question: ...
       A: option A
       B: option B
       C: option C
       D: option D
       Correct: ...
       Explanation: ...
       ```
     * Handles special cases ("All of the above", multi-choice sub-options).
     * Builds a metadata dict containing option texts, correct indices, subject/topic, and the combined text block.
  4. **Embedding Generation**:

     * Initializes `HuggingFaceEmbeddings` with `BAAI/bge-large-en-v1.5` (1024-dim).
     * Uses mixed-precision (`torch.amp.autocast`) on GPU if available.
     * Normalizes embeddings for cosine similarity.
  5. **Pinecone Upsert**:

     * Creates (or clears) a `medicalbot` index with dimension 1024 and cosine metric.
     * Batches of 64 vectors are upserted with their metadata.

* **Run**:

  ```bash
  python store_index.py --data_percent 1.0
  ```

  This indexes 100% of the train split. Adjust `--data_percent` for quicker tests.

### 2. Embeddings Strategy

* **Model Choice**: `BAAI/bge-large-en-v1.5`—a high-quality 1024-dim embedding optimized for English text.
* **Normalization**: Ensures that computing dot product is equivalent to cosine similarity.
* **Performance**: Mixed precision on GPU reduces memory footprint; CPU fallback is supported without modification.

### 3. LangGraph Pipeline

Defines a clear, directed sequence of operations:

1. **retrieve**: Dense vector search in Pinecone (top-5).
2. **conditional**: If top-1 score < `SIM_THRESH` (default 0.65), branch to **HyDE**; otherwise go to **rerank**.
3. **HyDE**: Generate a hypothetical answer via OpenAI, embed that text, and re-query Pinecone (covers cold-start).
4. **rerank**: Optionally use Cohere's reranker to reorder the retrieved snippets for relevance (skips if no key).
5. **refine**: Final ChatCompletion call combining system prompt, conversation history, and retrieved context. Limits to 3 sentences.

This modular graph (implemented with `StateGraph`) allows easy testing, monitoring, and replacement of individual nodes.

### 4. Fallback & Reranking

* **HyDE** (Hypothesize-then-Retrieve): Ensures the system never returns zero results.
* **Cohere Reranker**: Improves result ordering when a reranker key is provided; otherwise logs and skips.

### 5. FastAPI Service & Concurrency

* **FastAPI** for asynchronous HTTP & WebSocket endpoints.
* **Thread Pool Offload**: All CPU-bound calls (embeddings, Pinecone, OpenAI) run inside `run_in_threadpool` so the event loop remains responsive.
* **Endpoints**:

  * `/`: Bootstrap chat UI.
  * `/get` (POST): AJAX JSON chat.
  * `/ws`: WebSocket JSON chat.

### 6. Observability & Monitoring

* **Prometheus Client**:

  * `requests_total` counter for total chats.
  * `request_latency_seconds` histogram for `/get` latency.
  * `branch` counter (labels: retrieve, hyde, rerank, refine) to see pipeline usage.
* **Metrics Endpoint**: `/metrics` (raw Prometheus format) and styled `/metrics-ui`.
* **Health Check**: `/health` (raw JSON) and styled `/health-ui`.
* **Dashboard Spec**: JSON at `/dashboard.json` and styled UI at `/dashboard`—ready to import into Grafana or view in-browser.

### 7. Caching & Rate Limiting

* **TTLCache**: 10-minute default cache for repeat queries.
* **Sliding-Window Rate Limit**: 30 requests/min per IP to prevent abuse, returns HTTP 429 when exceeded.

---

## Installation & Setup

### Requirements & Environment Variables

Create a `.env` file with:

```
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY= your-openai-key
COHERE_API_KEY= your-cohere-key   # optional
PINECONE_ENVIRONMENT=us-east1-gcp
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
SIM_THRESH=0.65
CACHE_TTL=600
RATE_LIMIT=30
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Manual Run (Local)

1. **Create Conda Environment** *(recommended)*:

   ```bash
   conda env create -f environment.yml
   conda activate medbot
   ```

2. **Index** the dataset:

   ```bash
   python store_index.py --data_percent 1.0
   ```

3. **Start** the server:

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8080
   ```

4. **Browse**:

   * Chat UI: [http://localhost:8080/](http://localhost:8080/)
   * Health UI: [http://localhost:8080/health-ui](http://localhost:8080/health-ui)
   * Metrics UI: [http://localhost:8080/metrics-ui](http://localhost:8080/metrics-ui)
   * Dashboard: [http://localhost:8080/dashboard](http://localhost:8080/dashboard)

### Docker Build & Run

1. **Pin** your dependencies:

   ```bash
   pip freeze > requirements.txt
   ```

2. **Build**:

   ```bash
   docker build -t medbot:latest .
   ```

3. **Run** (replace keys):

   ```bash
   docker run -d -p 8080:8080 \
     -e PINECONE_API_KEY="…" \
     -e OPENAI_API_KEY="…" \
     -e COHERE_API_KEY="…" \
     medbot:latest
   ```

4. **Access** the same URLs as above (port 8080).

5. **(⭐ DockerHub Pull – coming soon)**

   *(The image is currently being uploaded and should be available around **July 12, 2025 16:00 GMT+2 time**. Due to limited bandwidth, upload is slow. Also, the Dockerfile is not yet optimized — expect a larger image size for now.)*

   Once ready, you can pull and run MedBot directly via:

   ```bash
   docker pull vishaals0507/medbot:latest
   docker run -d -p 8080:8080 \
     -e PINECONE_API_KEY="…" \
     -e OPENAI_API_KEY="…" \
     -e COHERE_API_KEY="…" \
     vishaals0507/medbot:latest
   ```

---

## Available Endpoints & UI

| Route             | Method    | Description                |
| ----------------- | --------- | -------------------------- |
| `/`               | GET       | Chat UI                    |
| `/get`            | POST      | AJAX chat (JSON response)  |
| `/ws`             | WebSocket | Real-time chat             |
| `/health`         | GET       | Raw health JSON            |
| `/health-ui`      | GET       | Styled health page         |
| `/metrics`        | GET       | Raw Prometheus metrics     |
| `/metrics-ui`     | GET       | Styled metrics page        |
| `/dashboard.json` | GET       | Raw Grafana dashboard JSON |
| `/dashboard`      | GET       | Styled dashboard JSON      |

---

## Reproducing the Indexing Pipeline

```bash
python store_index.py --data_percent 0.1
```

* Loads 10% of MedMCQA for quick feedback.
* Processes text, handles sub-options, explanation parsing.
* Generates mixed-precision embeddings in batches and upserts them to Pinecone.
* Due to time constraints and limited compute resources, the current embeddings was generated using only 10% of the data (during the Medbot build)

---

## Appendix: Configuration Options

* **`--data_percent`**: Fraction of the dataset to index (0.0–1.0).
* **`SIM_THRESH`**: Similarity threshold to trigger HyDE (default 0.65).
* **`CACHE_TTL`**: TTLCache duration in seconds (default 600).
* **`RATE_LIMIT`**: Max requests per minute per IP (default 30).

---

