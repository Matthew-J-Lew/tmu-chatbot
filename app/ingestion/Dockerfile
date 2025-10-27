FROM python:3.11-slim

# System deps for lxml, pdfminer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and allowlist
COPY ingest.py /app/ingest.py
COPY allowlist.yaml /app/allowlist.yaml

ENV PGHOST=pg PGUSER=rag PGPASSWORD=rag PGDATABASE=ragdb
# (Optionally set HTTP(S) proxy/certs here if needed on campus networks)

CMD ["python", "ingest.py"]
