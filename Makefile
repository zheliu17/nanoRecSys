# Makefile for nanoRecSys

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose

# Default arguments (can be overridden via command line: make train-retriever ARGS="--epochs 10")
ARGS :=

# Phony targets
.PHONY: help install format lint test data train-retriever mine-negatives train-ranker train-all serve stop clean clean-artifacts

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make install         Install dependencies in editable mode"
	@echo "  make data            Download and process the MovieLens dataset"
	@echo "  make train-retriever Train the retrieval model (Two-Tower)"
	@echo "  make mine-negatives  Mine hard negatives using the trained retriever"
	@echo "  make train-ranker    Train the ranking model (Cross-Encoder)"
	@echo "  make train-all       Run the full training pipeline (Data -> Retrieval -> Mining -> Ranking)"
	@echo "  make serve           Start the production serving stack (FastAPI, Redis, Streamlit)"
	@echo "  make stop            Stop the serving stack"
	@echo "  make test            Run unit and integration tests"
	@echo "  make clean           Remove validation logs and pycache"
	@echo "  make clean-artifacts Remove trained models and indices (WARNING: Deletes 'artifacts/*')"

# Installation
install:
	$(PIP) install -e .

# Data Processing
data:
	$(PYTHON) -m nanoRecSys.data.build_dataset
	$(PYTHON) -m nanoRecSys.data.splits

# Training Pipeline
train-retriever:
	$(PYTHON) -m nanoRecSys.train --mode retriever $(ARGS)

mine-negatives:
	$(PYTHON) -m nanoRecSys.training.mine_negatives

train-ranker:
	$(PYTHON) -m nanoRecSys.train --mode ranker $(ARGS)

train-all: data train-retriever mine-negatives train-ranker

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
