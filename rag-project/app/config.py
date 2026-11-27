import logging
import os

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("rag.config")


def _get(name, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    val = raw.strip()
    if cast is str:
        return val
    try:
        return cast(val)
    except (ValueError, TypeError):
        log.warning("Invalid env value for %s; using default %r", name, default)
        return default


EMBED_PROVIDER = _get("EMBED_PROVIDER", "cohere").lower()
if EMBED_PROVIDER not in ("cohere", "jina"):
    log.warning("Unknown EMBED_PROVIDER=%r; falling back to 'cohere'", EMBED_PROVIDER)
    EMBED_PROVIDER = "cohere"

COHERE_API_KEY = _get("COHERE_API_KEY", "")
JINA_API_KEY = _get("JINA_API_KEY", "")
GROQ_API_KEY = _get("GROQ_API_KEY", "")
QDRANT_URL = _get("QDRANT_URL", "")
QDRANT_API_KEY = _get("QDRANT_API_KEY", "")

# --- Providers / models ------------------------------------------------------
_cohere = EMBED_PROVIDER == "cohere"
EMBED_MODEL = _get("EMBED_MODEL", "embed-multilingual-v3.0" if _cohere else "jina-embeddings-v3")
EMBED_DIM = max(1, _get("EMBED_DIM", 1024, int))
RERANK_MODEL = _get("RERANK_MODEL", "rerank-v3.5" if _cohere else "jina-reranker-v2-base-multilingual")
LLM_MODEL = _get("LLM_MODEL", "openai/gpt-oss-120b")
QDRANT_COLLECTION = _get("QDRANT_COLLECTION", "mini_rag")

# --- Chunking (800-1200 tokens, 10-15% overlap per assessment) --------------
CHUNK_SIZE_TOKENS = max(100, _get("CHUNK_SIZE_TOKENS", 1000, int))
CHUNK_OVERLAP_TOKENS = max(0, min(_get("CHUNK_OVERLAP_TOKENS", 150, int), CHUNK_SIZE_TOKENS // 2))

# --- Retrieval ---------------------------------------------------------------
MMR_K = max(1, _get("MMR_K", 8, int))
RETRIEVE_K = max(MMR_K, _get("RETRIEVE_K", 20, int))
MMR_LAMBDA = max(0.0, min(1.0, _get("MMR_LAMBDA", 0.6, float)))
RERANK_TOP_N = max(1, _get("RERANK_TOP_N", 4, int))
MIN_RERANK_SCORE = max(0.0, _get("MIN_RERANK_SCORE", 0.15, float))

# --- Limits ------------------------------------------------------------------
MAX_DOC_CHARS = max(1000, _get("MAX_DOC_CHARS", 400_000, int))
MAX_UPLOAD_BYTES = max(10_000, _get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024, int))
MAX_QUESTION_CHARS = max(10, _get("MAX_QUESTION_CHARS", 1000, int))

# --- Rough price estimates, USD per 1M tokens (override if pricing changes) --
PRICE_EMBED_PER_M = max(0.0, _get("PRICE_EMBED_PER_M", 0.05, float))
PRICE_LLM_IN_PER_M = max(0.0, _get("PRICE_LLM_IN_PER_M", 0.59, float))
PRICE_LLM_OUT_PER_M = max(0.0, _get("PRICE_LLM_OUT_PER_M", 0.79, float))


def missing_keys():
    ranker_key = ("COHERE_API_KEY", COHERE_API_KEY) if _cohere else ("JINA_API_KEY", JINA_API_KEY)
    pairs = {
        ranker_key[0]: ranker_key[1],
        "GROQ_API_KEY": GROQ_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }
    return [k for k, v in pairs.items() if not v]