FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

FROM base AS runtime
RUN pip install --no-cache-dir .
RUN addgroup --system finvoice && adduser --system --ingroup finvoice --home /nonexistent finvoice

USER finvoice

EXPOSE 8000

CMD ["uvicorn", "finvoice_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS development
COPY tests ./tests
RUN pip install --no-cache-dir '.[dev]'

CMD ["pytest"]

FROM base AS asr
RUN pip install --no-cache-dir '.[asr]'
RUN addgroup --system finvoice && adduser --system --ingroup finvoice --home /nonexistent finvoice

USER finvoice

EXPOSE 8000

CMD ["uvicorn", "finvoice_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
