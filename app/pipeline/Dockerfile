FROM mcr.microsoft.com/playwright/python:v1.57.0-noble

# System deps for building wheels + SSL certs + python venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use a venv so installs don't touch system site-packages (more stable on WSL/Docker Desktop)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120

# Reuse the ingestion dependency set (includes embeddings model deps)
COPY app/ingestion/requirements.txt ./requirements.txt

# Install CPU-only torch first to avoid pulling large CUDA wheels from PyPI
RUN pip install --upgrade pip setuptools wheel \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY app /app/app
COPY app/crawler/profiles.yaml /app/profiles.yaml

ENV PGHOST=pg PGUSER=rag PGPASSWORD=rag PGDATABASE=ragdb
ENV CRAWL_PROFILES_YAML=/app/profiles.yaml

CMD ["python", "-m", "app.pipeline.scheduler"]
