# RESULTS — Task 4

## Live URL

- **API (FastAPI)**: https://hridaykedia-facedetectbackend.hf.space
  - `/health` returns `{"ok": true}`
  - `/docs` for interactive Swagger UI
- **UI (Streamlit)**: https://hridaykedia-facedetectfrontend.hf.space
  - Two tabs (Enroll / Search) that call the API above

## Setup

- **Embedding model**: InsightFace `antelopev2` — glintr100 ArcFace (ResNet-100 backbone, trained on Glint360K). 512-D L2-normalized embeddings. Picked over the more commonly-recommended `buffalo_l` because `antelopev2` uses the newer SCRFD-10G detector at the same accuracy class.
- **Face detector / aligner**: SCRFD-10G (bundled in `antelopev2`). It detects bbox + 5 landmarks (eyes, nose, mouth corners) and InsightFace internally applies a 2D similarity transform mapping those landmarks to a canonical 112×112 template before passing to the recognition net. `det_size=(320, 320)` instead of the default `(640, 640)` because the default loses faces in small (~95×125 px) LFW crops.
- **Qdrant**: Qdrant Cloud free tier, 1 GB cluster in AWS `us-east-1`. HNSW index, 512-D vectors, `Distance.COSINE`. Search via `client.query_points(...)` (the modern qdrant-client API, server-side ANN).

## Enrolled identities

5 identities from LFW (one photo each, downloaded by `starter-repo/scripts/get_sample_faces.py`):

| id  | source                                       |
|-----|----------------------------------------------|
| id0 | LFW — Atal Bihari Vajpayee                   |
| id1 | LFW — David Trimble                          |
| id2 | LFW — Donald Rumsfeld                        |
| id3 | LFW — Cherie Blair                           |
| id4 | LFW — John Negroponte                        |

Stored as 5 Qdrant points, each with `payload.identity_id = "idN"`. The endpoint supports multi-photo enrollment (`/enroll?id=alice` with multiple image fields stores N points sharing the same identity_id), exercised separately to confirm the multi-vector gallery pattern works end-to-end.

## Query result

- **Query image**: `sample/query_should_match_id0.jpg` (a different LFW photo of Atal Bihari Vajpayee)
- **Top match returned (deployed)**: `id0`
- **Cosine score (deployed)**: **0.6864**
- **Above threshold (0.40)**: yes
- **Correct?**: yes

Cross-check: the deployed cosine (0.6864405) and the local cosine (0.6864195) agree to four decimals — the pipeline is deterministic across machines, which would not be the case if alignment or normalization were inconsistent.

## Hard negative (look-alike)

Our 5 sample identities don't look alike (every cross-identity cosine is in [−0.09, +0.04]). To get a meaningful look-alike pair for the writeup, `scripts/pick_hard_negative.py` swept 120 LFW identities (~7000 cross-identity pairs) and picked the highest impostor cosine.

- **The two similar identities**: **George W. Bush vs. George H. W. Bush** (father and son)
- **Cosine score between them**: **0.2978**

Top-5 highest impostor cosines from the sweep (for context):

| rank | pair                                  | cosine |
|------|---------------------------------------|--------|
| 1    | George W. Bush, George H. W. Bush     | 0.2978 |
| 2    | Colin Powell, Hipolito Mejia          | 0.2438 |
| 3    | Richard Myers, Bruce Van De Velde     | 0.2434 |
| 4    | Brian Heidik, Tom Cruise              | 0.2433 |
| 5    | King Abdullah II, Michael Schumacher  | 0.2400 |

### Threshold — chosen empirically AND cross-checked against literature

**0.40**, justified two ways:

1. **Empirical on the canonical LFW 6000-pair verification protocol** (`scripts/compute_threshold_lfw_pairs.py` — same protocol every published ArcFace eval uses; 7669 unique images embedded):

   | | n | min | p5 | p50 | p95 | max |
   |---|---|---|---|---|---|---|
   | Genuine cosines (same identity) | 2979 | −0.008 | 0.509 | **0.680** | 0.825 | 0.953 |
   | Impostor cosines (different identities) | 2973 | −0.202 | −0.103 | 0.003 | 0.114 | **0.289** |

   FAR/FRR table on the same set:

   | threshold | FRR (genuine rejected) | FAR (impostor accepted) |
   |---|---|---|
   | 0.30 | 0.30 % | **0.00 %** |
   | 0.35 | 0.47 % | **0.00 %** |
   | **0.40** | **0.97 %** | **0.00 %** |
   | 0.45 | 2.11 % | 0.00 % |
   | 0.50 | 4.33 % | 0.00 % |

   The empirical impostor max across **2973 different-identity pairs** is **0.289** — meaning even at threshold 0.30, no impostor in the protocol clears the bar. 0.40 sits 0.11 above the worst observed impostor (a 38% safety margin) while costing only 0.97% FRR.

2. **Literature default**: ArcFace verification thresholds on LFW are published in the 0.35–0.45 range (Deng et al. 2019; InsightFace docs). At 0.40, published evals report FAR ≈ 0.01%, FRR ≈ 0.5%. Our empirical numbers are slightly tighter on FAR (literally 0 across our sample) and slightly looser on FRR — within noise of the published baseline.

## Search latency

Measured by `scripts/measure_latency.py` against the deployed URL (n=10, warmup dropped).

| metric | p50 | p95 |
|---|---|---|
| **Qdrant ANN call only** (server-reported `latency_ms`) | **17 ms** | **86 ms** |
| End-to-end (client → /search → client) | 2512 ms | 3281 ms |

The Qdrant number is what the brief asks for — server-side HNSW ANN search, fast because the backend Space and Qdrant Cloud are both in AWS `us-east-1` (~5–20 ms intra-region RTT).

The end-to-end number is dominated by **CPU embedding time on HF Spaces' free CPU tier** — antelopev2 detection + alignment + ArcFace inference takes ~2.4 s per face on free-tier CPU vs ~190 ms on an M-series laptop. A paid GPU Space would collapse that to <100 ms. We chose the free tier deliberately for the $0 cost; the latency story is honestly reported.

**Cold start**: ~10–15 s when the Space wakes from sleep (model is pre-baked into the Docker image, so no model download — just ONNX session initialization).

## Anything that broke / would improve with more time

**Real bugs we hit and fixed:**

- **`antelopev2` ships with a nested zip directory** (`~/.insightface/models/antelopev2/antelopev2/*.onnx`) which trips `'detection' in self.models`. Fixed with a `mv + rmdir` step both locally and baked into the Dockerfile build.
- **`det_size=(640, 640)` default loses faces in ~95×125 px LFW crops** because SCRFD's anchors are tuned for typical-sized photos. Dropped to `(320, 320)`; works for both LFW and larger inputs.
- **PIL doesn't apply EXIF orientation by default**, so phone selfies (sideways pixel bytes + a "rotate on display" EXIF tag) arrived sideways and SCRFD missed the face. Fixed with `ImageOps.exif_transpose()` in `embed()`. Validated end-to-end on a real phone selfie that now matches a passport photo of the same person with cosine **0.5978**.

**Would improve with more time:**

- **Adaptive `det_size` based on input image size** — current static `(320, 320)` would lose faces in full-body / hi-res inputs where the face occupies a small fraction of the canvas. ~5 lines to fix dynamically per request.
- **Fallback rotations** — for the small minority of images that arrive sideways without EXIF metadata, retry 90/180/270 rotations if first detection returns zero faces.
- **Quality scoring at enroll** — use `det_score` and face-area filters to reject low-quality gallery photos before they poison searches. ~20 lines.
- **Multi-level match output** (`definite_match` / `possible_match` / `no_match`) for applications that need a "send to human review" path instead of binary accept/reject.
- **GPU inference** — collapses end-to-end latency from 2.5 s to <100 ms; enables real-time use.
- **Anti-spoofing** — InsightFace's `silentface` model would catch photo-of-photo attacks. Required for any real biometric application.
