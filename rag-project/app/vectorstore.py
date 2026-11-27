"""Qdrant Cloud wrapper.

Collection schema
-----------------
vectors : single dense vector, size=EMBED_DIM (1024), distance=Cosine
point id: uuid5(namespace:source:position)  -> re-ingesting a document is idempotent
payload : text, source, title, section, position, tokens, namespace
indexes : keyword on `namespace` and `source` (needed for filtering on Qdrant Cloud)
"""
import uuid

from qdrant_client import QdrantClient, models

from . import config

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=30)
    return _client


def ensure_collection():
    c = client()
    if not c.collection_exists(config.QDRANT_COLLECTION):
        c.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE))
        for field in ("namespace", "source"):
            c.create_payload_index(config.QDRANT_COLLECTION, field, models.PayloadSchemaType.KEYWORD)


def _filter(namespace, source=None):
    must = [models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    return models.Filter(must=must)


def replace_document(namespace, source, title, chunks, vectors):
    """Upsert strategy: delete the document's old chunks, then upsert the new ones."""
    c = client()
    c.delete(config.QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=_filter(namespace, source)))
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{source}:{ch['position']}")),
            vector=vec,
            payload={"text": ch["text"], "source": source, "title": title, "section": ch["section"],
                     "position": ch["position"], "tokens": ch["tokens"], "namespace": namespace})
        for ch, vec in zip(chunks, vectors)
    ]
    c.upsert(config.QDRANT_COLLECTION, points=points, wait=True)


def search(vector, namespace, limit):
    return client().query_points(
        config.QDRANT_COLLECTION, query=vector, limit=limit, query_filter=_filter(namespace),
        with_payload=True, with_vectors=True).points


def list_sources(namespace):
    counts, offset = {}, None
    while True:
        pts, offset = client().scroll(config.QDRANT_COLLECTION, scroll_filter=_filter(namespace),
                                      limit=256, offset=offset, with_payload=["source", "title"])
        for p in pts:
            key = p.payload["source"]
            row = counts.setdefault(key, {"source": key, "title": p.payload.get("title", key), "chunks": 0})
            row["chunks"] += 1
        if offset is None:
            break
    return list(counts.values())


def clear_namespace(namespace):
    client().delete(config.QDRANT_COLLECTION,
                    points_selector=models.FilterSelector(filter=_filter(namespace)))
