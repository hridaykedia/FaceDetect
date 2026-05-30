"""
Task 4 — Face Match + Vector Search
Fill in the TODOs. Keep the endpoint contracts as-is.
"""
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Face Match + Vector Search")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll")
async def enroll(id: str, image: UploadFile = File(...)):
    """
    1. read image bytes
    2. detect + ALIGN the face   <-- do not skip alignment
    3. compute ArcFace/InsightFace embedding
    4. upsert into Qdrant with payload {"id": id}
    return: {"id": id, "stored": true}
    """
    # TODO
    return {"id": id, "stored": False, "note": "not implemented"}


@app.post("/search")
async def search(image: UploadFile = File(...)):
    """
    1. read image bytes
    2. detect + align + embed (same pipeline as enroll)
    3. Qdrant ANN search, top_k=1   <-- use ANN, not a python loop
    return: {"match_id": "...", "cosine": 0.0}
    """
    # TODO
    return {"match_id": None, "cosine": 0.0, "note": "not implemented"}
