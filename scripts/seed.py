"""Enroll the 5 sample identities by calling /enroll.

Usage:
    python scripts/seed.py                                # local default
    python scripts/seed.py --base-url https://your.host   # deployed
"""
import argparse
import glob
import sys

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sample-dir", default="sample")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.sample_dir}/id*.jpg"))
    if not paths:
        print(f"no images found in {args.sample_dir}/", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        # filename like sample/id0_Atal_Bihari_Vajpayee.jpg -> id "id0"
        identity = path.split("/")[-1].split("_", 1)[0]
        with open(path, "rb") as f:
            r = httpx.post(
                f"{args.base_url}/enroll",
                params={"id": identity},
                files={"image": (path, f, "image/jpeg")},
                timeout=60.0,
            )
        r.raise_for_status()
        print(f"  {identity:5s} <- {path}   {r.json()}")


if __name__ == "__main__":
    main()
