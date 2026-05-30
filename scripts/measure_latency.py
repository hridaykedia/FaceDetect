"""Measure /search latency. Drops first call (warm-up), reports p50/p95.

Usage:
    python scripts/measure_latency.py
    python scripts/measure_latency.py --base-url https://your.host --n 20
"""
import argparse
import statistics
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default="sample/query_should_match_id0.jpg")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    e2e_ms: list[float] = []
    search_ms: list[float] = []

    with open(args.image, "rb") as f:
        body = f.read()

    with httpx.Client(timeout=60.0) as client:
        for i in range(args.n + 1):
            t0 = time.perf_counter()
            r = client.post(
                f"{args.base_url}/search",
                files={"image": (args.image, body, "image/jpeg")},
            )
            t1 = time.perf_counter()
            r.raise_for_status()
            data = r.json()
            if i == 0:
                print(f"  warmup (dropped): e2e={1000*(t1-t0):.0f}ms  search={data['latency_ms']}ms")
                continue
            e2e_ms.append(1000 * (t1 - t0))
            search_ms.append(data["latency_ms"])
            print(f"  call {i:2d}: e2e={1000*(t1-t0):.0f}ms  search={data['latency_ms']}ms  match={data['match_id']}  cos={data['cosine']:.4f}")

    print()
    print(f"e2e (client → /search → client):  p50={statistics.median(e2e_ms):.0f}ms  p95={sorted(e2e_ms)[int(0.95*len(e2e_ms))]:.0f}ms")
    print(f"search (Qdrant ANN only):         p50={statistics.median(search_ms):.0f}ms  p95={sorted(search_ms)[int(0.95*len(search_ms))]:.0f}ms")


if __name__ == "__main__":
    main()
