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
   docker compose run --rm api python -m app.tools.inspect_prompt "QUESTION"

   note that k and num_candidates can be tweaked
2. Milestone 2 is complete when we can call retrieve(query, k) and it gives us the best pages/chunks for the question


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

IMPORTANT TUNING KNOBS:
app/ingestion/ingest.py
CHUNK_TOKEN_SIZE
CHUNK_TOKEN_OVERLAP

TESTING: This should print the full exact prompt sent to LLM as well as retrieved context:
docker compose run --rm api python -m app.tools.inspect_prompt "Can you list all the undergraduate and undergraduate programs?"

This will just show important chunks:
docker compose run --rm api python -m app.tools.inspect_prompt "QUESTION"

RUNNING FULL PIPELINE:

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Can you list all the undergraduate programs?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Tell me about the co-op program"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is Mohamed Lachemi?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is the dean of the entire school? not just the faculty of arts?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is Dr Amy Peng?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is the dean of the entire school?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who created you?"} | ConvertTo-Json) 
| ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is the IT lead of the faculty of 
arts?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is Michael MacDonald?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is Valerie Deacon?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is the leader of the faculty of arts?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who can I go to if i have additional 
questions?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="List a few students from the dean's list in 2021"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="What webpages can i go to to read about the co-op program?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Who is Matthew Lu?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" -ContentType "application/json" -Body (@{question="Can you list all the undergraduate programs?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10


BONUS MILESTONE: WEB SPIDER/CRAWLING  + DATABASE INGESTION 
1. We give the spider a starting page/seed (in this case https://www.torontomu.ca/arts/)
2. It downloads the page and looks for links connected to it
3. Checks the collected links against rules
   - allowed website domains
   - allowed paths
   - allowed patterns + regex
4. Save results into a database, approved, blocked, failed
5. Process repeats for approved pages
Commands:
1. start services:
   - docker composen up -d --build pg redis ollama api
2. Run crawler
   - docker compose --profile crawl run -rm --build crawler
3. Run ingestion
docker compose --profile ingest run --rm --build ingestion `
  python -m app.ingestion.ingest --mode db --profile arts --limit 200

Change the limit as you see fit

Look into using selenium or scrapy instead of beautiful soup (has problems ingesting javascript/drop downs)