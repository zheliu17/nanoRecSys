# Reproducibility

This document explains how to reproduce the main `nanoRecSys` pipeline and artifacts.

There are **two supported reproducibility paths**:

1. **Fast artifact reproduction** using published pretrained checkpoints.
2. **Full end-to-end retraining** starting from raw MovieLens data.

The first path is intended for reviewers or users who want to get the system running quickly.
The second path is intended for users who want to rebuild the full training pipeline from scratch.

---

## Reproducibility Philosophy

This repository is designed so that users do **not** need to retrain the sequential retriever just to understand or run the serving stack.

To make the project easier to verify:

- pretrained retriever checkpoints are published externally,
- `pipeline.py` supports skipping retriever training,
- the pipeline can then rebuild downstream artifacts deterministically from those checkpoints,
- and the serving stack can be launched locally from generated artifacts.

This is a deliberate design choice: full retriever training is expensive, while most downstream verification tasks are not.

---

## Path A: Fast Reproduction Using Published Checkpoints

This is the recommended path for most users.

### What this reproduces

This path reproduces:

- processed data,
- item/user embedding generation,
- FAISS index build,
- hard-negative mining,
- ranker training,
- ONNX export,
- and local serving.

### Why this path exists

The retriever is the most expensive training stage in the repo.
To reduce setup time, the pipeline supports skipping retriever training and downloading published weights if they are not already present.

From `pipeline.py`, when `--skip-retriever` is used, the pipeline:

- checks for `item_tower.pth`
- checks for `user_tower.pth`
- downloads them if missing
- then continues the rest of the pipeline

### Steps

```bash
git clone https://github.com/zheliu17/nanoRecSys.git
cd nanoRecSys
pip install -e .[all]
make data
python pipeline.py run --skip-retriever
```

After this finishes, you should have the artifacts needed for local serving.

To launch the stack:

```bash
make serve
```

---

## Path B: Full End-to-End Retraining

This path rebuilds the full system from raw data.

### What this reproduces

This path runs:

1. data processing
2. retriever training
3. embedding generation
4. FAISS index build
5. hard-negative mining
6. ranker training
7. ONNX export

### Steps

```bash
git clone https://github.com/zheliu17/nanoRecSys.git
cd nanoRecSys
pip install -e .[all]
make data
python pipeline.py run
```

A Makefile-based equivalent is:

```bash
make data
make train-retriever
make post-train
```

---

## Pipeline Stages

The main pipeline stages are:

1. **process_data**
2. **train_retriever**
3. **build_index**
4. **mine_negatives**
5. **train_ranker**
6. **export_onnx**

### Stage 1: Data processing

This stage runs:

- raw dataset processing,
- chronological train/val/test split generation,
- prebuilt sequential file generation.

### Stage 2: Retriever

This stage either:

- trains the sequential retriever from scratch, or
- skips training and downloads pretrained `item_tower.pth` and `user_tower.pth`.

### Stage 3: Index build

This stage materializes:

- item embeddings,
- user embeddings,
- FAISS index artifacts.

### Stage 4: Hard-negative mining

This stage mines difficult negatives from retriever outputs for ranker training.

### Stage 5: Ranker training

This stage trains the lightweight second-stage reranker.

### Stage 6: ONNX export

This stage converts the serving models into ONNX artifacts used by the optimized inference path.

---

## Default Main-Path Training Configuration

The main pipeline uses the following core defaults for the sequential retriever:

- user tower type: `transformer`
- epochs: `300`
- batch size: `128`
- learning rate: `1e-3`
- num workers: `4`

The main ranker stage uses:

- epochs: `5`
- batch size: `2048`
- learning rate: `1e-3`
- random negative ratio: `0.01`
- warmup steps: `500`

Hard-negative mining defaults:

- `top_k=100`
- `skip_top=10`
- `sampling_ratio=0.2`

These values come directly from the checked-in pipeline and Makefile targets.

---

## What “reproducible” means in this project

In this repository, reproducibility means:

- a reviewer can rebuild the main serving artifacts from code,
- the main pipeline is scripted rather than manually assembled,
- expensive checkpoints are publishable and reusable,
- and the online stack can be stood up locally from those artifacts.

It does **not** necessarily mean that every training run will produce byte-for-byte identical metrics across every machine or environment.

Factors that may change exact results include:

- hardware differences,
- library versions,
- random seeds,
- nondeterministic GPU kernels,
- and optional workflow differences between notebook and script-based execution.

For that reason, this project distinguishes between:

- **functional reproducibility**: can the system be rebuilt and run?
- **numerical identity**: do all runs match exactly?

The repository is optimized primarily for the first, while keeping the second as reasonable as possible.

---

## Notebook Reproduction

The repo also includes notebooks for:

- sequential transformer training,
- static baseline embeddings,
- LLM reranker training.

These are intended for interactive exploration and walkthroughs.

---

## LLM Reranker Reproduction

The LLM reranker has a separate workflow and heavier hardware requirements.

To prepare data for the LLM reranker, the docs and notebook assume that:

- retriever checkpoints already exist,
- item embeddings can be generated,
- hard negatives can be mined with the `llm_ranker` suffix.

This is a distinct experimental path from the default online serving pipeline.

The LLM reranker is therefore best understood as:

- reproducible as a research extension,
- but not part of the default low-latency serving path.

---

## Recommended Reviewer Workflow

If you are reviewing the project and want the shortest path to verification:

1. install the repo,
2. run `make data`,
3. run `python pipeline.py run --skip-retriever`,
4. launch with `make serve`,
5. inspect the generated artifacts and latency behavior.

This path verifies the end-to-end engineering workflow without requiring the most expensive training stage.

---
