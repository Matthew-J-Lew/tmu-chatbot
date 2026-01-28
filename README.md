MILESTONE 0 SETUP:
1. ssh/login in to a TMU virtual machine/new general machine
2. install docker and docker compose plugin
   sudo apt update
   sudo apt install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker
3. clone the repository
   cd /opt
   sudo git clone https://github.com/Matthew-J-Lew/tmu-chatbot.git
   cd into the project file
4. Retrieve environment variables
5. build and start containers
   docker compose up -d --build
6. verify that containers are running
   docker ps
   you should see pg, ollama, and api listed and healthy
7. pull the model
   docker compose exec ollama ollama pull llama3.1:8b
8. confirm model is installed
   docker compose exec ollama ollama list
9. milestone 0 is complete when docker ps shows postgres, ollama, and api all healthy and responding

MILESTONE 1 INGESTION PIPELINE:

1. ensure postgres and ollama containers are running and healthy
   docker ps
2. verify schema.sql is mounted correctly in /app/ingestion and tables exist
   docker compose exec pg psql -U rag -d ragdb -c "\dt"
3. confirm the schema contains the sources and chunks tables
   docker compose exec pg psql -U rag -d ragdb -c "\d+ sources"
   docker compose exec pg psql -U rag -d ragdb -c "\d+ chunks"
4. build the ingestion image
   docker compose build ingestion
5. run the ingestion pipeline (reads from /app/allowlist.yaml)
   docker compose --profile ingest up --no-deps ingestion
6. verify chunk count after ingestion
   docker compose exec pg psql -U rag -d ragdb -c "SELECT COUNT(*) FROM chunks;"
   the count should be 50+ after ingesting 3–5 official TMU pages
7. optionally view sample data
   docker compose exec pg psql -U rag -d ragdb -c "SELECT * FROM v_chunks_basic LIMIT 5;"
8. milestone 1 is complete when we can ingest chunks from all the sources in the allowlist.yaml

MILESTONE 1.5 OPTIONAL VISUALIZATION:

1. to view the database in a GUI, use Adminer or pgAdmin
2. for Adminer, add this service to docker-compose.yml
   adminer:
   image: adminer:latest
   restart: always
   ports:
   - 8080:8080
   depends_on:
   - pg
3. start it with docker compose up -d adminer
4. access Adminer at [http://localhost:8080](http://localhost:8080)
   system: PostgreSQL
   server: pg
   username: rag
   password: rag
   database: ragdb
5. you can browse, filter, and export data through the web interface
6. milestone 1.5 is complete when you can visualize the chunks table and confirm embeddings exist

MILESTONE 2 RETRIEVAL AND RERANKING (RAG SYSTEM)
1. Build the API image to pick up requirements + code changes
   docker compose build api
2. Start postgres, Ollama, and API
   docker compose up -d pg ollama api
3. Verify containers
   docker ps
   confirm that pg, ollama, and api are running
TESTING:
1. Open a shell inside the API container
   docker compose run --rm api bash
   - You can then ask questions with:
   python -c "from app.rag.debug import debug_retrieve; debug_retrieve('INSERT QUESTION HERE', k=6, num_candidates=30)"
   note that k and num_candidates can be tweaked
2. Milestone 2 is complete when we can call retrieve(query, k) and it gives us the best pages/chunks for the question
3. 
python - << 'EOF'
import asyncio, os, asyncpg
from app.rag.retrieval import retrieve

async def main():
    pool = await asyncpg.create_pool(
        dsn=os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=5,
    )
    try:
        results = await retrieve(
            pool,
            "What undergraduate programs are offered at the faculty of arts?",
            k=6,
            num_candidates=30,
        )
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['source_url']}")
            print(r['chunk'][:30000])
    finally:
        await pool.close()

asyncio.run(main())
EOF


MILESTONE 3 LLM CALL (MVP of the whole system)
1. Ensure all containers are running:
- docker compose up -d --build
- docker ps
2. Test the health of the endpoint
- curl http://localhost:8000/healthz
- It should return {"status":"ok"}
3. Test the chat endpoint (this is powershell syntax):
- (Invoke-WebRequest -Uri "http://localhost:8000/api/chat" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"question": "YOUR QUESTION GOES HERE"}' `
).Content
4. If the OLLAMA LLM didn't get installed properly, use:
docker compose exec ollama ollama pull llama3.1:8b

MILESTONE 3.5 PULLING DIFFERENT LLMS:
1. Pull the model
- docker compose exec ollama ollama pull <model_name>
2. Verify it's installed
- docker compose exec ollama ollama list
3. in docker-compose.yml change OLLAMA_MODEL:<model_name>
4. restart the container
- docker compose up -d --build (optional: api)

