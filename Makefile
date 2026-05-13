# Makefile for the "Better Embeddings for Better GNNs" project.
# Targets are intentionally thin: each delegates to a Python entry point or
# a one-liner shell command, so behavior stays the same on Linux, macOS, and
# Windows (under git-bash / WSL).

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

CONFIG ?=

.PHONY: install test lint run-experiment

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -q tests

lint:
	$(RUFF) check src tests scripts

# Usage: make run-experiment CONFIG=experiments/h1_tfidf_gcn.yaml
run-experiment:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Usage: make run-experiment CONFIG=experiments/<config>.yaml"; \
		exit 2; \
	fi
	$(PYTHON) -m scripts.run_experiment --config $(CONFIG)
