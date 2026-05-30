"""Empirical threshold on the canonical LFW verification protocol.

fetch_lfw_pairs('10_folds') gives the 6000 verification pairs LFW
ships for benchmarking: 3000 genuine (same identity, different photo)
and 3000 impostor (different identities). This is the same protocol
published ArcFace evals use, so the numbers here are directly
comparable to the literature.

Caches embeddings by image hash because LFW pairs reuse images
heavily (~5700 unique from 12000 pair-images).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings import embed  # noqa: E402


def main():
    print("Loading LFW pairs (10_folds subset, 6000 pairs)...")
    pairs = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0)
    print(f"  {len(pairs.pairs)} pairs loaded ({int(pairs.target.sum())} genuine, {int((1 - pairs.target).sum())} impostor)")

    cache: dict[bytes, np.ndarray] = {}
    misses = 0

    def emb_cached(arr: np.ndarray) -> np.ndarray | None:
        nonlocal misses
        img = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        key = img.tobytes()
        if key in cache:
            return cache[key]
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        try:
            e = embed(buf.getvalue())
            cache[key] = e
            return e
        except Exception:
            misses += 1
            return None

    genuine: list[float] = []
    impostor: list[float] = []
    t0 = time.perf_counter()

    for i, ((a_img, b_img), label) in enumerate(zip(pairs.pairs, pairs.target)):
        a = emb_cached(a_img)
        b = emb_cached(b_img)
        if a is None or b is None:
            continue
        cos = float(np.dot(a, b))
        (genuine if label == 1 else impostor).append(cos)

        if (i + 1) % 250 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs.pairs) - i - 1) / rate
            print(f"  [{i+1:>4d}/{len(pairs.pairs)}]  unique embeds={len(cache)}  misses={misses}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\nDone in {time.perf_counter() - t0:.0f}s.  unique embeds={len(cache)}  detection misses={misses}")

    def pct(xs, p):
        return float(np.percentile(xs, p))

    print()
    print(f"Genuine cosines  (n={len(genuine):4d}):  min={min(genuine):.4f}  p5={pct(genuine,5):.4f}  p50={pct(genuine,50):.4f}  p95={pct(genuine,95):.4f}  max={max(genuine):.4f}")
    print(f"Impostor cosines (n={len(impostor):4d}):  min={min(impostor):.4f}  p5={pct(impostor,5):.4f}  p50={pct(impostor,50):.4f}  p95={pct(impostor,95):.4f}  max={max(impostor):.4f}")

    suggested = round((pct(genuine, 5) + max(impostor)) / 2, 2)
    print()
    print(f"Suggested empirical threshold: {suggested:.2f}")
    print(f"  (midpoint of impostor max {max(impostor):.4f} and genuine p5 {pct(genuine,5):.4f})")
    print(f"Literature default: ~0.40 (ArcFace verification on LFW)")

    # FAR/FRR at different thresholds
    g = np.array(genuine)
    im = np.array(impostor)
    print()
    print(f"{'threshold':>10s}  {'FRR (genuine rejected)':>22s}  {'FAR (impostor accepted)':>23s}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
        frr = (g < t).mean()
        far = (im >= t).mean()
        print(f"{t:>10.2f}  {frr*100:>20.2f}%  {far*100:>21.2f}%")

    with open("threshold_data_lfw_pairs.json", "w") as f:
        json.dump(
            {
                "genuine_cosines": genuine,
                "impostor_cosines": impostor,
                "genuine_stats": {"min": min(genuine), "p5": pct(genuine, 5), "p50": pct(genuine, 50), "p95": pct(genuine, 95), "max": max(genuine)},
                "impostor_stats": {"min": min(impostor), "p5": pct(impostor, 5), "p50": pct(impostor, 50), "p95": pct(impostor, 95), "max": max(impostor)},
                "suggested_threshold": suggested,
                "literature_threshold": 0.40,
                "n_pairs_protocol": len(pairs.pairs),
                "unique_images_embedded": len(cache),
                "detection_misses": misses,
            },
            f,
            indent=2,
        )
    print("\nRaw cosines + stats written to threshold_data_lfw_pairs.json")


if __name__ == "__main__":
    main()
