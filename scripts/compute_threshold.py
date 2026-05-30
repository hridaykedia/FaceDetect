"""Empirical genuine vs impostor cosine distribution on our 5 sample identities.

Pulls multiple LFW photos per identity, embeds them, computes:
  - genuine cosines (same person, different photo)
  - impostor cosines (different people)
Prints percentiles + suggests a threshold in the gap.
Dumps raw cosines to threshold_data.json for the RESULTS.md table.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings import embed  # noqa: E402

# Names of the 5 identities the starter sample script picked.
SAMPLE_NAMES = [
    "Atal Bihari Vajpayee",
    "David Trimble",
    "Donald Rumsfeld",
    "Cherie Blair",
    "John Negroponte",
]


def _embed_pil(img: np.ndarray) -> np.ndarray | None:
    """Embed an LFW image array (HxWxC, [0,1] floats). Returns None if no face."""
    arr = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    try:
        return embed(buf.getvalue())
    except Exception:
        return None


def main():
    data = fetch_lfw_people(min_faces_per_person=5, color=True, resize=1.0)
    name_to_idx = {n: i for i, n in enumerate(data.target_names)}

    embeddings: dict[str, list[np.ndarray]] = {}
    for name in SAMPLE_NAMES:
        if name not in name_to_idx:
            print(f"  WARN: {name} has <5 photos in LFW, skipping")
            continue
        tid = name_to_idx[name]
        photos = data.images[data.target == tid]
        vecs = []
        for img in photos[:6]:  # cap at 6 per identity to keep it fast
            v = _embed_pil(img)
            if v is not None:
                vecs.append(v)
        embeddings[name] = vecs
        print(f"  {name}: {len(vecs)} embeddings")

    genuine, impostor = [], []
    for name, vecs in embeddings.items():
        for a, b in combinations(vecs, 2):
            genuine.append(float(np.dot(a, b)))
    names = list(embeddings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in embeddings[names[i]]:
                for b in embeddings[names[j]]:
                    impostor.append(float(np.dot(a, b)))

    def pct(xs, p):
        return float(np.percentile(xs, p))

    print()
    print(f"Genuine cosines  (n={len(genuine):3d}):  min={min(genuine):.4f}  p5={pct(genuine,5):.4f}  p50={pct(genuine,50):.4f}  p95={pct(genuine,95):.4f}  max={max(genuine):.4f}")
    print(f"Impostor cosines (n={len(impostor):3d}):  min={min(impostor):.4f}  p5={pct(impostor,5):.4f}  p50={pct(impostor,50):.4f}  p95={pct(impostor,95):.4f}  max={max(impostor):.4f}")

    suggested = round((pct(genuine, 5) + max(impostor)) / 2, 2)
    print()
    print(f"Suggested empirical threshold: {suggested:.2f}")
    print(f"  (midpoint of impostor max {max(impostor):.4f} and genuine p5 {pct(genuine,5):.4f})")
    print(f"Literature default: ~0.40 (ArcFace verification on LFW)")

    with open("threshold_data.json", "w") as f:
        json.dump(
            {
                "genuine_cosines": genuine,
                "impostor_cosines": impostor,
                "genuine_stats": {"min": min(genuine), "p5": pct(genuine, 5), "p50": pct(genuine, 50), "p95": pct(genuine, 95), "max": max(genuine)},
                "impostor_stats": {"min": min(impostor), "p5": pct(impostor, 5), "p50": pct(impostor, 50), "p95": pct(impostor, 95), "max": max(impostor)},
                "suggested_threshold": suggested,
                "literature_threshold": 0.40,
            },
            f,
            indent=2,
        )
    print("Raw cosines + stats written to threshold_data.json")


if __name__ == "__main__":
    main()
