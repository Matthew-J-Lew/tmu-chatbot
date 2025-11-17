FROM python:3.11-slim

WORKDIR /app

# Copy requirements for the API service
COPY app/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app package (api, rag, ingestion, etc.)
COPY app /app/app

# Optionally keep your existing main entrypoint
COPY app/api/main.py /app/main.py

# Default command for the API container
CMD ["python", "main.py"]
