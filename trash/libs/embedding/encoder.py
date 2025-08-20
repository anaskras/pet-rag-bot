import os
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _model = SentenceTransformer(name)
    return _model

def encode_texts(texts: list[str]) -> list[list[float]]:
    m = get_model()
    vecs = m.encode(texts, normalize_embeddings=True).tolist()
    return vecs

def dim() -> int:
    return get_model().get_sentence_embedding_dimension()
