"""HTTP clients for embedding, reranking, and chat providers."""
import logging

import httpx

from . import config

log = logging.getLogger("rag.providers")
_http = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))


class ProviderError(RuntimeError):
    """Raised with a user-safe message when an upstream provider fails."""


def _post(name: str, url: str, key: str, payload: dict):
    if not key:
        raise ProviderError(f"{name} API key is not configured on the server.")
    try:
        response = _http.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
    except httpx.HTTPError as exc:
        log.error("%s network error: %s", name, exc)
        raise ProviderError(f"Could not reach {name}. Try again in a moment.") from exc

    if response.status_code == 429:
        raise ProviderError(f"{name} rate limit reached. Wait a few seconds and retry.")
    if response.status_code >= 400:
        log.error("%s HTTP %s: %s", name, response.status_code, response.text[:300])
        raise ProviderError(f"{name} returned an error ({response.status_code}).")
    try:
        return response.json()
    except ValueError as exc:
        log.error("%s returned invalid JSON", name)
        raise ProviderError(f"{name} returned an invalid response. Try again.") from exc


def embed(texts: list[str], task: str):
    """Embed document or query texts; return vectors and the provider token count."""
    vectors, tokens = [], 0
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        if config.EMBED_PROVIDER == "cohere":
            data = _post(
                "Cohere embeddings",
                "https://api.cohere.com/v2/embed",
                config.COHERE_API_KEY,
                {
                    "model": config.EMBED_MODEL,
                    "texts": batch,
                    "embedding_types": ["float"],
                    "input_type": "search_query" if task.endswith("query") else "search_document",
                },
            )
            vectors.extend(data["embeddings"]["float"])
            tokens += data.get("meta", {}).get("billed_units", {}).get("input_tokens", 0) or 0
        else:
            data = _post(
                "Jina embeddings",
                "https://api.jina.ai/v1/embeddings",
                config.JINA_API_KEY,
                {
                    "model": config.EMBED_MODEL,
                    "task": task,
                    "dimensions": config.EMBED_DIM,
                    "input": batch,
                },
            )
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in rows)
            tokens += data.get("usage", {}).get("total_tokens", 0)
    return vectors, tokens


def rerank(query: str, documents: list[str], top_n: int):
    """Return provider rerank indices and relevance scores."""
    if config.EMBED_PROVIDER == "cohere":
        name, url, key = (
            "Cohere reranker",
            "https://api.cohere.com/v2/rerank",
            config.COHERE_API_KEY,
        )
    else:
        name, url, key = (
            "Jina reranker",
            "https://api.jina.ai/v1/rerank",
            config.JINA_API_KEY,
        )
    data = _post(
        name,
        url,
        key,
        {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
    )
    return [
        {"index": row["index"], "score": row["relevance_score"]}
        for row in data["results"]
    ]


def chat(system: str, user: str):
    """Generate a grounded response with Groq; return text and token counts."""
    data = _post(
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        config.GROQ_API_KEY,
        {
            "model": config.LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    usage = data.get("usage", {})
    return (
        data["choices"][0]["message"]["content"].strip(),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )
