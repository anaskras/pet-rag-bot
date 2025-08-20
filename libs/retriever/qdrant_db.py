import os, sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# Добавляем путь к корню проекта
sys.path.append(Path(__file__).parent.parent.parent.__str__())

from libs.embedding.encoder import encode_texts, dim


def client():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def ensure_collection(name: str):
    c = client()
    if name not in [col.name for col in c.get_collections().collections]:
        c.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim(), distance=Distance.COSINE),
        )


def upsert(collection: str, payloads: list[dict]):
    c = client()
    vectors = encode_texts([p["text"] for p in payloads])
    points = [PointStruct(id=p["id"], vector=vectors[i], payload=p) for i, p in enumerate(payloads)]
    c.upsert(collection_name=collection, points=points)


def search(collection: str, query: str, limit=5, filters=None):
    vec = encode_texts([query])[0]
    res = client().search(collection_name=collection, query_vector=vec, limit=limit, query_filter=filters)
    # возвратим payload, чтобы был text/url/section/ids
    return [hit.payload for hit in res]


