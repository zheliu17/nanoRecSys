# Makefile for nanoRecSys

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
RANKER_ARGS := --mode ranker \
	--user_tower_type transformer \
	--epochs 5 \
	--batch_size 2048 \
	--random_neg_ratio 0.01 \
	--lr 1e-3 \
	--item_lr 0.0 \
	--num_workers 2 \
	--warmup_steps 500 \
	--check_val_every_n_epoch 1
MINING_ARGS := nanoRecSys.training.mine_negatives_sasrec --batch_size 128 --top_k 100 --skip_top 10 --sampling_ratio 0.2

# Default arguments (can be overridden via command line: make train-retriever ARGS="--epochs 10")
ARGS :=

# Phony targets
.PHONY: help install format lint test data train-ckpt post-train mine-negatives train-ranker train-all serve stop clean clean-artifacts

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make install         Install dependencies in editable mode"
	@echo "  make data            Download and process the MovieLens dataset"
	@echo "  make train-ckpt      Train retriever checkpoint (hard-coded args)"
	@echo "  make post-train      Run remaining steps after training (embeddings, mining, ranker, index)"
	@echo "  make mine-negatives  Mine hard negatives using the trained retriever"
	@echo "  make train-ranker    Train the ranking model (Cross-Encoder)"
	@echo "  make build-index     Build FAISS index for retrieval"
	@echo "  make serve           Start the production serving stack (FastAPI, Redis, Streamlit)"
	@echo "  make stop            Stop the serving stack"
	@echo "  make test            Run unit and integration tests"
	@echo "  make clean           Remove validation logs and pycache"
	@echo "  make clean-artifacts Remove trained models and indices (WARNING: Deletes 'artifacts/*')"

# Installation
install:
	$(PIP) install -e .[all]

# Data Processing
data:
	$(PYTHON) -m nanoRecSys.data.build_dataset --task process
	$(PYTHON) -m nanoRecSys.data.splits
	$(PYTHON) -m nanoRecSys.data.build_dataset --task prebuild

train-retriever:
	$(PYTHON) -m nanoRecSys.train \
		--mode retriever \
		--user_tower_type transformer \
		--epochs 300 \
		--batch_size 128 \
		--lr 1e-3 \
		--num_workers 4 \
		$(ARGS)

mine-negatives:
	$(PYTHON) -m $(MINING_ARGS)

train-ranker:
	$(PYTHON) -m nanoRecSys.train $(RANKER_ARGS) $(ARGS)

build-index:
	$(PYTHON) -m nanoRecSys.indexing.build_faiss_flat

post-train:
	$(PYTHON) -m nanoRecSys.indexing.build_embeddings --mode all
	$(PYTHON) -m nanoRecSys.indexing.build_faiss_flat
	$(PYTHON) -m $(MINING_ARGS)
	$(PYTHON) -m nanoRecSys.train $(RANKER_ARGS) $(ARGS)

# Serving
serve:
	$(DOCKER_COMPOSE) up --build

stop:
	$(DOCKER_COMPOSE) down

# Testing
test:
	pytest tests/

# Cleaning
clean:
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-artifacts:
	rm -rf artifacts/*
