.PHONY: format lint run test

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
