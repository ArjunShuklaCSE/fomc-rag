FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
WORKDIR /app

# dependencies first (cached layer); Linux resolves torch from the CPU wheel index (see pyproject.toml)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY results ./results
COPY data/manifest.jsonl ./data/manifest.jsonl
RUN uv sync --locked --no-dev

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# First start: build the corpus (download + clean, ~5 min) unless a prepared data/ volume is mounted.
# The serving index is built on first startup and cached in data/index/.
CMD ["sh", "-c", "[ -f data/processed/docs.jsonl ] || uv run --no-dev python -m fomcrag.ingest; exec uv run --no-dev uvicorn fomcrag.api:app --host 0.0.0.0 --port 8000"]
