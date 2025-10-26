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
