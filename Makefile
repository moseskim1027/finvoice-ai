.PHONY: evaluate-retrieval format lint run test

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

test:
	pytest
