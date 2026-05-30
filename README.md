# Face Match + Vector Search

FastAPI service that detects + aligns faces with InsightFace `antelopev2` (SCRFD-10G detector + glintr100 ArcFace recognition), embeds them as 512-D L2-normalized vectors, and stores + searches them in **Qdrant Cloud** via HNSW ANN. A single embedding path is used by both `/enroll` and `/search` so the spaces stay comparable.

## Live demos

- **UI**: https://hridaykedia-facedetectfrontend.hf.space
  Streamlit two-tab app (Enroll / Search) calling the API above.
- **API (this Space)**: https://hridaykedia-facedetectbackend.hf.space/docs
  Interactive Swagger docs at `/docs`, health at `/health`.

## Endpoints

- `GET /health` → `{"ok": true}`
- `POST /enroll?id=<identity>` — multipart `image` (one or more) → `{"id", "stored"}`
- `POST /search` — multipart `image` → `{"match_id", "cosine", "above_threshold", "latency_ms"}`

`/enroll` accepts multiple images for the same `id`; each becomes a Qdrant point sharing `payload.identity_id`, and `/search` groups the top-k results by identity (multi-vector gallery pattern).

## Headline results

- Query match (LFW): top match `id0`, cosine **0.6864**, correct ✓
- Threshold: **0.40**, justified empirically on the canonical LFW 6000-pair verification protocol (FAR = 0.00%, FRR = 0.97%) and matches the ArcFace literature default
- Hard negative: George W. Bush vs. George H. W. Bush, cosine **0.2978** → correctly rejected
- Deployed search latency (Qdrant ANN call): **17 ms p50**, **86 ms p95**

Full numbers, reproduction, and threshold/FAR-FRR table in [`RESULTS.md`](RESULTS.md). AI-development log in [`AI_LOG.md`](AI_LOG.md).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# .env with your Qdrant Cloud credentials
echo "QDRANT_URL=https://<your-cluster>.aws.cloud.qdrant.io:6333" > .env
echo "QDRANT_API_KEY=..." >> .env

# Download LFW sample faces
python starter-repo/scripts/get_sample_faces.py

# Start the API
uvicorn app.main:app --reload

# (optional) seed the deployed URL the same way
python scripts/seed.py --base-url http://127.0.0.1:8000
```

## Deploy

Two Hugging Face Spaces — backend (Docker SDK, this repo) and frontend (Streamlit SDK, `frontend/`). Both fit on the free CPU tier.

**Backend Space (FastAPI)**
1. Create a Space (SDK: Docker, hardware: CPU basic).
2. Settings → Variables and secrets: set `QDRANT_URL` and `QDRANT_API_KEY`.
3. Push the repo root to the Space's git remote — but first prepend the HF Spaces frontmatter to `README.md` (HF requires it; it's stripped from this repo's README so GitHub renders cleanly):

   ```yaml
   ---
   title: Face Match Vector Search
   emoji: 🔍
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

**Frontend Space (Streamlit)** — see [`frontend/`](frontend/)
1. Create a Space (SDK: Streamlit, hardware: CPU basic).
2. Settings → Variables and secrets: set `BACKEND_URL` to the backend Space URL.
3. Push `frontend/` as the root of the Streamlit Space (e.g. via `git subtree push --prefix=frontend space-frontend main`).

## Repository layout

```
app/                                  # FastAPI service
  embeddings.py                       # single embed() pipeline (antelopev2)
  vector_store.py                     # Qdrant Cloud HNSW wrapper
  main.py                             # /health, /enroll, /search
  config.py                           # env var loading
frontend/                             # Streamlit UI (separate HF Space)
  app.py                              # two-tab Enroll/Search UI
scripts/                              # local helpers
  seed.py                             # enroll the 5 sample identities
  measure_latency.py                  # p50/p95 against any base URL
  compute_threshold.py                # 5-identity empirical threshold
  compute_threshold_lfw_pairs.py      # canonical LFW 6000-pair protocol
  pick_hard_negative.py               # find highest-cosine impostor pair
starter-repo/                         # the task brief's starter (untouched)
Dockerfile                            # HF Spaces backend deploy
RESULTS.md                            # measured numbers
AI_LOG.md                             # AI development log
```
