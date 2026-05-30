"""Find a real look-alike (hard-negative) pair in LFW.

Our 5 sample identities don't look alike — all cross-identity cosines are
near zero. To get a meaningful "hard negative" for the writeup, we sweep
a wider LFW slice and pick the highest-cosine impostor pair.

Saves the two crops side-by-side as sample/hard_negative.jpg and prints
the cosine + identity names.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings import embed  # noqa: E402


def _embed_lfw(img_arr: np.ndarray) -> np.ndarray | None:
    arr = (img_arr * 255).astype(np.uint8) if img_arr.max() <= 1.0 else img_arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    try:
        return embed(buf.getvalue())
    except Exception:
        return None


def main():
    # Pull a wider set: 2+ photos per person → ~1700 identities
    data = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
    images, targets, names = data.images, data.target, data.target_names

    # One representative photo per identity (cap at 120 identities → ~7k pairs, manageable)
    LIMIT = 120
    pick_idx_by_target: dict[int, int] = {}
    for i, t in enumerate(targets):
        if t not in pick_idx_by_target:
            pick_idx_by_target[t] = i
        if len(pick_idx_by_target) >= LIMIT:
            break

    chosen = list(pick_idx_by_target.items())
    print(f"Embedding {len(chosen)} identities (one photo each) ...")
    embeddings = []
    for tid, idx in chosen:
        v = _embed_lfw(images[idx])
        if v is not None:
            embeddings.append((tid, idx, v))
    print(f"  {len(embeddings)} valid embeddings (rest had no detectable face)")

    # All pairwise cosines, keep top 5
    top: list[tuple[float, int, int, int, int]] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            cos = float(np.dot(embeddings[i][2], embeddings[j][2]))
            top.append((cos, embeddings[i][0], embeddings[i][1], embeddings[j][0], embeddings[j][1]))
    top.sort(key=lambda x: -x[0])

    print()
    print("Top-5 highest cross-identity cosines (visual look-alikes):")
    for cos, t1, i1, t2, i2 in top[:5]:
        print(f"  cos={cos:.4f}  {names[t1]:30s}  vs  {names[t2]}")

    # Save the #1 pair as a side-by-side image
    best = top[0]
    cos, t1, i1, t2, i2 = best
    a = images[i1]
    b = images[i2]
    a = (a * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    b = (b * 255).astype(np.uint8) if b.max() <= 1.0 else b.astype(np.uint8)
    h = max(a.shape[0], b.shape[0])
    side = np.zeros((h, a.shape[1] + b.shape[1] + 8, 3), dtype=np.uint8)
    side[: a.shape[0], : a.shape[1]] = a
    side[: b.shape[0], a.shape[1] + 8 :] = b
    out_path = "sample/hard_negative.jpg"
    Image.fromarray(side).save(out_path)
    print()
    print(f"Saved side-by-side to {out_path}")
    print(f"Hard negative: {names[t1]} vs {names[t2]}   cosine = {cos:.4f}")
    print(f"With threshold = 0.40, this pair is {'ACCEPTED (false match!)' if cos >= 0.40 else 'REJECTED (correct)'}")


if __name__ == "__main__":
    main()
