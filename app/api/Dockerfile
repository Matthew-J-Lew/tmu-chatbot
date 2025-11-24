FROM python:3.11-slim

WORKDIR /app

# Copy requirements for the API service
COPY app/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app package (api, rag, ingestion, etc.)
COPY app /app/app

# Copy main.py to the top level so we can refer to it as 'main:app'
COPY app/api/main.py /app/main.py

# Start the FastAPI app using Uvicorn (the ASGI web server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
