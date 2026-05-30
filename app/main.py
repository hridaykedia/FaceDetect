from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import config, embeddings, vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store.ensure_collection()
    embeddings.get_app()  # warm the model so first request isn't slow
    yield


app = FastAPI(title="Face Match + Vector Search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll")
async def enroll(id: str, image: List[UploadFile] = File(...)):
    """Enroll one or more images under a single identity id.

    Each uploaded image becomes a separate Qdrant point sharing
    `payload.identity_id == id` — multi-vector gallery pattern.
    """
    stored = 0
    for upload in image:
        data = await upload.read()
        try:
            vector = embeddings.embed(data)
        except embeddings.NoFaceDetected:
            raise HTTPException(status_code=400, detail=f"no face detected in {upload.filename}")
        vector_store.upsert(id, vector)
        stored += 1
    return {"id": id, "stored": stored}


@app.post("/search")
async def search(image: UploadFile = File(...)):
    data = await image.read()
    try:
        vector = embeddings.embed(data)
    except embeddings.NoFaceDetected:
        raise HTTPException(status_code=400, detail="no face detected")

    start = time.perf_counter()
    hits = vector_store.search(vector, top_k=5)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if not hits:
        return {"match_id": None, "cosine": None, "above_threshold": False, "latency_ms": latency_ms}

    best_by_identity: dict[str, float] = {}
    for identity_id, score in hits:
        if score > best_by_identity.get(identity_id, -1.0):
            best_by_identity[identity_id] = score
    match_id, cosine = max(best_by_identity.items(), key=lambda kv: kv[1])

    return {
        "match_id": match_id,
        "cosine": cosine,
        "above_threshold": cosine >= config.THRESHOLD,
        "latency_ms": latency_ms,
    }
