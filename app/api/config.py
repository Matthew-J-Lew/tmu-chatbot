import os


def _get_csv(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    parts = [p.strip() for p in val.split(",")]
    return [p for p in parts if p]


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


# ---- DB ----
PGHOST = os.getenv("PGHOST", "pg")
PGPORT = _get_int("PGPORT", 5432)
PGDATABASE = os.getenv("PGDATABASE", "ragdb")
PGUSER = os.getenv("PGUSER", "rag")
PGPASSWORD = os.getenv("PGPASSWORD", "rag")

# ---- Redis ----
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL_RESPONSE = _get_int("CACHE_TTL_RESPONSE", 21600)   # 6h
CACHE_TTL_RETRIEVAL = _get_int("CACHE_TTL_RETRIEVAL", 21600) # 6h

# ---- RAG tuning ----
RAG_TOP_K = _get_int("RAG_TOP_K", 4)
RAG_NUM_CANDIDATES = _get_int("RAG_NUM_CANDIDATES", 12)
RERANK_ENABLED = _get_bool("RERANK_ENABLED", True)

# ---- Prompt limits ----
MAX_CHUNK_CHARS = _get_int("MAX_CHUNK_CHARS", 800)
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 3200)

# ---- CORS ----
CORS_ALLOW_ORIGINS = _get_csv("CORS_ALLOW_ORIGINS", ["https://www.torontomu.ca"])

# ---- LLM provider routing ----
# Choose which backend to use for generation.
# - "azure"  => Azure OpenAI (recommended for production)
# - "ollama" => Local Ollama container (fallback / offline dev)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Optional: if set, and the primary provider fails, try this provider once.
# Example: LLM_PROVIDER=azure, LLM_FALLBACK_PROVIDER=ollama
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower() or None

# ---- Ollama / LLM ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = _get_int("OLLAMA_TIMEOUT_SECONDS", 90)
OLLAMA_MAX_RETRIES = _get_int("OLLAMA_MAX_RETRIES", 2)

OLLAMA_NUM_PREDICT = _get_int("OLLAMA_NUM_PREDICT", 512)
OLLAMA_TEMPERATURE = _get_float("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_TOP_P = _get_float("OLLAMA_TOP_P", 0.9)

# ---- Azure OpenAI ----
# These defaults follow the manager-provided integration note.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://arts-chatbot-openai.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

# In Azure, the SDK's "model" parameter is the *deployment name*.
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Cost controls (recommended): keep <= 512
AZURE_OPENAI_MAX_TOKENS = min(_get_int("AZURE_OPENAI_MAX_TOKENS", 512), 512)

_tmp_temp = _get_float("AZURE_OPENAI_TEMPERATURE", 0.1)
AZURE_OPENAI_TEMPERATURE = max(0.0, min(2.0, _tmp_temp))

AZURE_OPENAI_TIMEOUT_SECONDS = _get_int("AZURE_OPENAI_TIMEOUT_SECONDS", 60)

# ---- Concurrency ----
MAX_CONCURRENT_LLM = _get_int("MAX_CONCURRENT_LLM", 2)
