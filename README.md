[![CI Pipeline](https://github.com/zheliu17/nanoRecSys/actions/workflows/ci.yml/badge.svg)](https://github.com/zheliu17/nanoRecSys/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# nanoRecSys: Production-Style Sequential Recommender System

**nanoRecSys** is an end-to-end recommender system project designed to reflect real production constraints (latency, caching, indexing, and operational workflows). It aims to bridge research ideas (**modern sequential retrieval**) and practical serving (CPU-friendly inference + vector search).

<https://github.com/user-attachments/assets/6d0c4713-c8ab-43ba-b197-d2602244cf35>

## Key Features

* **Modern Retrieval Architecture:** A **Sequential Transformer** (SASRec-based) enhanced with RoPE, SwiGLU, and InfoNCE Loss.
  * **0.287 HR@10** on MovieLens-20M, outperforming standard SASRec baselines by ~40% and matching optimized implementations (e.g., BERT4Rec).
* **High-Throughput Serving:** ONNX inference engine (FastAPI + AsyncIO) backed by Redis (caching) and **FAISS** (vector search).
* **Production Engineering:** A complete Docker Compose orchestration, CI/CD workflows, and a Streamlit frontend.
* **Full-Lifecycle Implementation:** From raw data processing and offline training (PyTorch) to online serving and latency benchmarking.

## Evaluation & Benchmarks

*See [Training.md](./docs/Training.md) for architectural deep-dives and [Benchmark.md](./docs/Benchmark.md) for latency analysis.*

#### Offline Performance

Compared against recent literature (ICML'24, WWW'25), our retrieval model performs competitively:

| Metric | Our Model | Standard SASRec | SOTA (HSTU/FuXi-α) |
| :--- | :--- | :--- | :--- |
| **HR@10** | **0.287** | ~0.20 - 0.29 | ~0.33 |

#### Online Latency

Load tested with `locust` on a CPU-only laptop setup:

* **Cold-path inference:** ~23ms (2 vCPU; transformer user embedding + FAISS + ranking)
* **P95 Latency:** 180ms (end-to-end)

## Quick Start

### 1. Installation

*Default installation includes training components only. Use [all] for serving dependencies.*

```bash
git clone https://github.com/zheliu17/nanoRecSys.git
cd nanoRecSys

# Install dependencies (Virtual Environment recommended)
make install  # (Equivalent to pip install -e .[all])
```

### 2. Training & Artifact Generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zheliu17/nanoRecSys/blob/main/docs/sequential_transformer.ipynb)

```bash
make data # Download and preprocess MovieLens-20M dataset
```

* **Train from scratch (~10 hrs):** See [Sequential Transformer Notebook](./docs/sequential_transformer.ipynb)

```bash
# export WANDB_MODE=offline
make train-retriever
# Or, download pretrained weights
# git clone https://huggingface.co/zheliu97/nanoRecSys artifacts
```

### 3. Ranker Training, Indexing, and Serving

*Also see [Sequential Transformer Notebook](./docs/sequential_transformer.ipynb)*

```bash
make post-train
make serve # (Equivalent to docker-compose up --build)
```

## System Architecture

```mermaid
graph TD
    User([User / Client]) -->|HTTP Request| API[FastAPI Gateway]

    subgraph "Serving Layer"
        API -->|1. Check Cache| Redis[(Redis Cache)]
        Redis -- Cache Hit --> API
        Redis -- Cache Miss --> Retrieval

        subgraph "Inference Pipeline"
            Retrieval[Retrieval Service] -->|2. Encode User Seq| QueryEnc[Transformer User Encoder]
            QueryEnc -->|3. Vector Search| FAISS[(FAISS Index)]
            FAISS -->|4. Candidates| Reranker[MLP Ranker]
            Reranker -->|5. Top-K Items| Redis
        end
    end

    subgraph "Offline Training"
        Data[(MovieLens Data)] --> Trainer[Training Pipeline]
        Trainer -->|Updates| QueryEnc
        Trainer -->|Updates| Reranker
        Trainer -->|Builds| FAISS
    end
```

## Project Structure

```text
.
├── artifacts/             # Trained models & indices (GitIgnored)
├── data/                  # Dataset storage
├── docs/                  # Training & Analysis documentation
├── frontend/              # Streamlit UI
├── serving/               # FastAPI Inference Server
├── src/
│   └── nanoRecSys/        # Core Library
├── tests/                 # Unit & Integration Tests
└── docker-compose.yml     # Orchestration
```
