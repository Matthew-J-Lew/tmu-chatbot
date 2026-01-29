FROM python:3.11-slim

# System deps for lxml/bs4 parser speed and robustness
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install crawler dependencies
COPY app/crawler/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared app package (crawler + ingestion + rag + tools)
COPY app /app/app

# Default crawl profile definitions (can be overridden by bind mount)
COPY app/crawler/profiles.yaml /app/profiles.yaml

ENV PGHOST=pg PGUSER=rag PGPASSWORD=rag PGDATABASE=ragdb
ENV CRAWL_PROFILES_YAML=/app/profiles.yaml

# Default command runs the 'arts' profile, bootstrapping it if needed
CMD ["python", "-m", "app.crawler.crawl", "--profile", "arts"]
