MILESTONE 0 SETUP:
1. ssh into the TMU virtual machine
2. install docker and docker compose plugin
   sudo apt update
   sudo apt install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker
3. clone the repository
   cd /opt
   sudo git clone [https://github.com/](https://github.com/)<your-org-or-username>/arts-ai-chatbot.git
   cd arts-ai-chatbot
4. copy environment file
   cp .env.example .env
5. build and start containers
   docker compose up -d --build
6. verify that containers are running
   docker ps
   you should see pg, ollama, and api listed and healthy
7. pull the model
   docker compose exec ollama ollama pull llama3.1:8b
8. confirm model is installed
   docker compose exec ollama ollama list
9. check health of API
   curl [http://localhost:8000/healthz](http://localhost:8000/healthz)
10. milestone 0 is complete when docker ps shows postgres, ollama, and api all healthy and responding

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
8. milestone 1 is complete when 50+ chunks are stored and verified

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

