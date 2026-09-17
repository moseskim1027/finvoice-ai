.PHONY: audit-speech evaluate-retrieval evaluate-speech format lint run run-mcp test

SPEECH_MANIFEST ?= data/speech-manifest.json
SPEECH_DATASET_ROOT ?= data
SPEECH_EVALUATION_ARGS ?= --split development

audit-speech:
	python -m finvoice_ai.evaluation.speech_integrity $(SPEECH_MANIFEST) --dataset-root $(SPEECH_DATASET_ROOT)

evaluate-retrieval:
	python -m finvoice_ai.evaluation.retrieval

evaluate-speech:
	python -m finvoice_ai.evaluation.speech_runner $(SPEECH_MANIFEST) --dataset-root $(SPEECH_DATASET_ROOT) $(SPEECH_EVALUATION_ARGS)

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .

run:
	uvicorn finvoice_ai.main:app --reload

run-mcp:
	python -m finvoice_ai.mcp_server

test:
	pytest
