# System Design

`nanoRecSys` is an end-to-end recommendation system built around a **two-stage retrieval + ranking architecture**.

The project is designed to reflect practical production constraints:

- low-latency online serving on CPU,
- offline training and artifact generation,
- vector retrieval over precomputed item embeddings,
- cache-backed inference for repeated requests,
- and a clean separation between model training and serving-time inference.

The main online path is:

1. receive a recommendation request,
2. check Redis for cached results,
3. encode the user's recent interaction sequence into a user embedding,
4. retrieve candidate items from a FAISS index,
5. rerank candidates with a lightweight ranker,
6. cache and return the top-K results.

The main offline path is:

1. process MovieLens-20M,
2. train or download retriever checkpoints,
3. build item/user embeddings,
4. build a FAISS index,
5. mine hard negatives,
6. train the reranker,
7. export serving artifacts to ONNX.

---

## High-Level Architecture

### Online serving path

- **FastAPI API** handles online inference requests.
- **Redis** stores cached recommendation results for repeated queries.
- **Retriever** encodes user history into a dense embedding.
- **FAISS** performs nearest-neighbor candidate retrieval over item embeddings.
- **MLP ranker** reranks retrieved candidates using embedding and metadata features.
- **Streamlit frontend** provides a lightweight demo UI.

### Offline training path

- raw MovieLens data is processed into training-ready sequence data,
- the retriever is trained as a sequential transformer,
- item and user embeddings are materialized,
- a FAISS index is built,
- hard negatives are mined from retriever outputs,
- a reranker is trained on positives + hard negatives + random negatives,
- ONNX export generates CPU-friendly inference artifacts.

---

## Component Breakdown

## 1. Data processing

Data processing converts MovieLens interactions into the inputs needed for sequence modeling and ranking.

From the training pipeline:

- `process_data()`
- `create_user_time_split(val_k=1, test_k=1)`
- `prebuild_sequential_files()`

This creates the processed sequence datasets used by the retriever and the later negative-mining / ranker stages.

### Design choice

The project uses a chronological split rather than random splitting. This is closer to realistic recommendation evaluation because future interactions are not leaked into training.

---

## 2. Retriever

The first-stage retriever is a **sequential transformer** that maps user history to a dense embedding and scores items via embedding similarity.

Default configuration includes:

- embedding dimension: `256`
- max sequence length: `200`
- transformer layers: `4`
- attention heads: `8`
- dropout: `0.2`
- positional embedding type: `RoPE`
- activation: `SwiGLU`
- loss: `InfoNCE` / in-batch retrieval training by default

The retriever is trained via `nanoRecSys.train` with `mode="retriever"`.

### Why a two-tower retriever?

A two-tower retrieval architecture separates:

- **user encoding** at query time
- **item encoding** offline

This enables fast ANN search because item embeddings can be precomputed and indexed once, while only the user representation must be computed online.

### Why a sequential transformer?

A recommender based only on static user IDs is easy to train, but misses temporal intent. The sequential transformer uses recent interaction history to model short-term preference shifts, which is more realistic for recommendation serving.

---

## 3. Embeddings and FAISS index

After retriever training, the system materializes:

- item embeddings,
- user embeddings,
- and a FAISS index over the item tower output.

The default serving configuration uses a **flat index**.

This is an intentional choice for the MovieLens-20M item corpus in this project: at ~27k items, exact search remains fast enough and avoids ANN recall tradeoffs. Separate benchmarking in `docs/Benchmark.md` explores when approximate indexing becomes more useful.

### Design choice

The system keeps indexing as a separate artifact-generation stage rather than recomputing embeddings online. This matches real production systems, where index build / refresh is typically an offline or nearline workflow.

---

## 4. Negative mining

The ranker is not trained only on positives vs random negatives. It also uses **hard negatives** mined from the retriever.

Pipeline defaults:

- `top_k=100`
- `skip_top=10`
- `sampling_ratio=0.2`

The intuition is:

- retrieve plausible but incorrect candidates from the current retriever,
- skip the very top items to avoid over-hard or noisy examples,
- train the ranker to separate true next items from near-miss candidates.

This creates a more realistic second-stage ranking problem than random negative sampling alone.

---

## 5. Ranker

The second-stage ranker is a lightweight MLP-based model that scores retrieved candidates using more than just embedding similarity.

Features used by the ranker include:

- user embedding,
- item embedding,
- genre multihot features,
- year features,
- normalized popularity,
- and optional ID dropout during training.

Training data mixes multiple supervision sources:

- positives,
- explicit negatives,
- hard negatives,
- random negatives.

This makes the reranker more expressive than pure vector similarity while remaining far cheaper than an LLM-based online model.

### Why keep the ranker lightweight?

The project explicitly optimizes for CPU-friendly serving. A lightweight reranker offers a strong latency/quality tradeoff and is suitable for the real-time online path, unlike the experimental LLM reranker.

---

## 6. ONNX export and serving

After training, models are exported to ONNX for serving.

The optimized production path in the repo uses:

- ONNX Runtime,
- async serving,
- FAISS retrieval,
- Redis caching.

This reduces inference overhead compared with raw PyTorch eager execution and supports the CPU-constrained latency benchmarks documented in `docs/Benchmark.md`.

---

## 7. Experimental LLM reranker

The repository also includes an experimental multimodal LLM reranker.

This component is intentionally **not** part of the default online serving stack.

Instead, it serves as a research extension exploring whether collaborative filtering item embeddings can be injected into a small LLM for candidate reranking.

Its role in the repository is:

- demonstrate research breadth,
- compare local fine-tuned LLM reranking against zero-shot LLM baselines,
- and illustrate why an LLM may improve offline ranking quality but still be a poor fit for a strict low-latency serving budget.

---

## Artifact Flow

The main artifacts produced by the system are:

- processed sequence data,
- retriever checkpoints:
  - `item_tower.pth`
  - `user_tower.pth`
- materialized embeddings,
- FAISS index files,
- negative-mining datasets,
- ranker checkpoints,
- ONNX export artifacts.

These artifacts are stored under the repo’s configured `artifacts/` and `data/` directories and are mounted into the serving container via Docker Compose.

---

## Deployment Topology

The default local deployment uses Docker Compose with three services:

- `api`
- `frontend`
- `redis`

### `api`

Responsible for serving inference requests. It mounts local `artifacts/` and `data/` into the container and uses Redis for caching.

### `frontend`

A Streamlit demo UI that talks to the API service through `API_URL=http://api:8000`.

### `redis`

A lightweight cache service used by the API layer.

---

## Tradeoffs and Non-Goals

## Tradeoffs

### Exact vs approximate retrieval

For the current item scale, the system defaults to flat FAISS search because exact retrieval is still fast and avoids ANN recall loss.

### Quality vs latency

The online serving path favors a sequential retriever + lightweight ranker over an LLM reranker. This is a deliberate tradeoff for CPU-friendly latency.

### Offline complexity for online simplicity

The pipeline invests more work offline:

- embedding generation,
- index building,
- negative mining,
- ONNX export

so that online inference stays simple and fast.

## Non-goals

This project does **not** currently aim to provide:

- online learning from live user feedback,
- real-time incremental index updates,
- multi-region deployment,
- canary rollout / shadow deployment tooling,
- large-scale production monitoring infrastructure,
- or a GPU-served LLM reranker in the real-time path.

Those are natural future extensions, but they are outside the current scope.

---

## Why this architecture?

This design intentionally mirrors a realistic recommendation stack:

- **retrieval** for scalable candidate generation,
- **ranking** for better precision,
- **offline artifact generation** for serving efficiency,
- **cache + ANN + ONNX** for latency,
- and a separate **experimental research branch** for more advanced modeling ideas.

The goal of the project is not just to train a recommender model, but to demonstrate the full engineering lifecycle required to move from offline modeling to production-style serving.
