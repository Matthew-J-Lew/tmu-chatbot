FROM python:3.11-slim

# System deps for lxml/pdf parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Reuse the ingestion dependency set (includes embeddings model deps)
COPY app/ingestion/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared app package
COPY app /app/app

# Default crawl profile definitions (can be overridden by bind mount)
COPY app/crawler/profiles.yaml /app/profiles.yaml

ENV PGHOST=pg PGUSER=rag PGPASSWORD=rag PGDATABASE=ragdb
ENV CRAWL_PROFILES_YAML=/app/profiles.yaml

# Run the scheduler loop by default
CMD ["python", "-m", "app.pipeline.scheduler"]
