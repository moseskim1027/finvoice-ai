.PHONY: audit-speech evaluate-retrieval evaluate-speech format lint observable-stack production-evidence research-baselines run run-mcp test

SPEECH_MANIFEST ?= data/speech-manifest.json
SPEECH_DATASET_ROOT ?= data
SPEECH_EVALUATION_ARGS ?= --split development
RESEARCH_OUTPUT ?= reports/multimodal-intent-baselines.json

audit-speech:
	python -m finvoice_ai.evaluation.speech_integrity $(SPEECH_MANIFEST) --dataset-root $(SPEECH_DATASET_ROOT)

evaluate-retrieval:
	python -m finvoice_ai.evaluation.retrieval

evaluate-speech:
	python -m finvoice_ai.evaluation.speech_runner $(SPEECH_MANIFEST) --dataset-root $(SPEECH_DATASET_ROOT) $(SPEECH_EVALUATION_ARGS)

research-baselines:
	python -m finvoice_ai.research.experiment --manifest $(SPEECH_MANIFEST) --dataset-root $(SPEECH_DATASET_ROOT) --development-asr-report reports/development-base-beam1.json --test-asr-report reports/test-base-beam1.json --output $(RESEARCH_OUTPUT)

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .

observable-stack:
	docker compose up --build api

production-evidence:
	python -m finvoice_ai.evaluation.production_benchmark --output src/finvoice_ai/evaluation/data/results/production_evidence.json

run:
	uvicorn finvoice_ai.main:app --reload

run-mcp:
	python -m finvoice_ai.mcp_server

test:
	pytest
