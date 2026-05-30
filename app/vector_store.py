from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config

_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def ensure_collection() -> None:
    if not _client.collection_exists(config.COLLECTION_NAME):
        _client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )


def upsert(identity_id: str, vector: np.ndarray) -> str:
    point_id = str(uuid.uuid4())
    _client.upsert(
        collection_name=config.COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"identity_id": identity_id},
            )
        ],
    )
    return point_id


def search(vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """ANN search via Qdrant HNSW. Returns [(identity_id, cosine), ...] desc by score."""
    response = _client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector.tolist(),
        limit=top_k,
    )
    return [(p.payload["identity_id"], p.score) for p in response.points]
