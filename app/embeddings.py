from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis


class NoFaceDetected(Exception):
    pass


_app: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    global _app
    if _app is None:
        app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
        # det_size=(640,640) misses faces in small (~250px) LFW images because
        # SCRFD's anchors are tuned for typical-sized photos in a 640 canvas.
        # 320 handles both small (LFW) and full-resolution inputs.
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _app = app
    return _app


def embed(image_bytes: bytes) -> np.ndarray:
    """Decode → detect → align → 512-D L2-normalized embedding (largest face)."""
    pil = Image.open(io.BytesIO(image_bytes))
    # Apply EXIF rotation — phone photos store sideways pixels + a "rotate on display"
    # tag. Without this, a portrait selfie comes in as landscape and SCRFD misses the face.
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    faces = get_app().get(bgr)
    if not faces:
        raise NoFaceDetected("no face detected")

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding
