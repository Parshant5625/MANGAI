FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./
COPY data ./data
COPY models ./models

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["sh", "-c", "python scripts/seed_demo.py --skip-train && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
