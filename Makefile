.PHONY: evaluate-retrieval format lint run run-mcp test

evaluate-retrieval:
	python -m finvoice_ai.evaluation.retrieval

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
