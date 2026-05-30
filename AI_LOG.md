# AI_LOG — Task 4

AI tools are allowed and encouraged. We want to see how you use them, not whether you avoided them.

## Tools used

- **Claude Code (Sonnet 4.6, 1M context)** — primary driver for code, the Dockerfile, scripts, and writeup drafts. Ran in plan-mode first to scope architecture decisions before any code, then in normal mode for implementation.
- **Qdrant Cloud web UI** — for cluster + API key creation (no AI involved).
- **Hugging Face Spaces web UI** — for Space creation + secret setup.

## How I used Claude Code (plan mode + decision discussions)

The most valuable part of the workflow wasn't any single prompt — it was using **Claude Code's plan mode** before writing any code. In plan mode Claude is read-only (no edits, no commits) except for a single plan document, so the entire architecture had to be argued out and written down before implementation started. The plan file became the contract; only after I approved it did Claude switch to normal mode and start writing code.

Four substantive discussions during plan mode shaped the whole project:

- **Embedding model** — I rejected the default suggestion (`buffalo_l`) and pushed for the broader field. `antelopev2` (SCRFD-10G + glintr100 ArcFace) came up only after I asked for alternatives, and we picked it for the newer detector at equivalent accuracy.
- **Hosting** — compared HF Spaces, Cloud Run, Fly.io, AWS EC2 free tier, and Railway against the timebox + rubric. HF Spaces won on free-tier RAM headroom for antelopev2 and zero deploy friction; EC2 ruled out on CPU-credit throttling + manual HTTPS setup eating into the timebox.
- **Threshold strategy** — chose "empirical AND literature" so the hard-negative writeup could cite measured numbers from our own pipeline *and* show alignment with the published ArcFace defaults.
- **Frontend architecture** — when I asked about deploying a Streamlit UI alongside the API, Claude initially suggested a single-container supervisord setup; I pushed back, we mapped HF Spaces' one-container-per-Space constraint, and landed on two separate Spaces (API + UI) as the cleanest fit.

Every architectural decision in this project has a planned discussion behind it, not a guessed default.

## Where the AI was WRONG / gave broken output, and how I caught it

- **Claude defaulted to recommending `buffalo_l`** as the embedding model — what every tutorial uses. I had to explicitly ask "what are the better alternatives" before `antelopev2` came up. Caught only because I pushed back; Claude did not proactively surface tradeoffs until prompted.
- **Suggested `det_size=(640, 640)` in `embeddings.py`** — the InsightFace default. Silently returned **zero faces** on LFW's 94×125 px crops because SCRFD's anchors are tuned for a face occupying a normal fraction of a 640 canvas; the upscaled LFW face becomes too small relative to the anchors. Caught by my first smoke test raising `NoFaceDetected`. Fix: `det_size=(320, 320)`.
- **Did not proactively flag the `antelopev2` nested-directory zip bug** — the `antelopev2` zip extracts as `~/.insightface/models/antelopev2/antelopev2/*.onnx` instead of flat, causing the `'detection' in self.models` assertion. Claude knew nothing about it until we hit it. Found by running the smoke test, inspecting the directory layout, and `mv + rmdir`-ing manually. The fix is now baked into the Dockerfile build (three-step pre-bake: download → flatten → verify).
- **Initial threshold script tried to embed LFW images directly from the sklearn array** without re-encoding to JPEG. This worked in isolation but produced subtly different cosines than running through the real `embed()` pipeline (which decodes JPEG bytes). Caught by spot-checking and rewriting to pass JPEG bytes through `embed()` — keeps the pre-processing identical to the production endpoint.
- **Missed EXIF orientation handling** — `PIL.Image.open()` doesn't apply the EXIF rotation tag by default, so a phone selfie with EXIF orientation 5 came in with sideways pixel bytes and SCRFD missed the face entirely. Claude didn't preempt this; caught when I tested with a real phone photo from my camera roll. Fix: `ImageOps.exif_transpose()` before BGR conversion.
- **Initial latency estimate was wrong** — Claude claimed `antelopev2` would take ~150 ms/face on CPU; the actual measurement on HF Spaces free CPU was ~2.4 s/face. The estimate was for an M-series laptop, not a shared free-tier CPU. Caught by running `measure_latency.py` against the deployed URL. RESULTS.md reports the real number honestly.

## Design decisions (2–3 lines each, why)

- **Embedding model: `antelopev2` (InsightFace).** SCRFD-10G detector + glintr100 ArcFace (ResNet-100, Glint360K), 512-D. Same accuracy class as `buffalo_l` with a newer SCRFD detector (faster, slightly better on small faces). HF Spaces' 16 GB RAM means there was no reason to drop down to `buffalo_sc`.

- **Alignment approach: InsightFace's built-in SCRFD 5-landmark + similarity-transform warp to 112×112.** Used `FaceAnalysis.get()` end-to-end, never manually cropped a bbox — that's the most common silent failure mode for ArcFace pipelines. Picked the **largest face by bbox area** when multiple are detected.

- **Threshold: 0.40, justified empirically + by literature.** Empirically: on the canonical LFW 6000-pair verification protocol (2979 genuine + 2973 impostor cosines through our pipeline), impostor max = **0.289**, genuine p5 = **0.509** — at threshold 0.40, **FAR = 0.00%, FRR = 0.97%**. Literature: ArcFace papers report 0.35–0.45 with FAR ≈ 0.01% at 0.40. Both converge. Erred to the stricter side because false-accept (silent security breach) is worse than false-reject (user retries) for identity verification.

- **Qdrant Cloud + cosine + L2-normalized vectors.** Used `face.normed_embedding` everywhere (unit length verified empirically). Qdrant collection configured with `Distance.COSINE`, 512-D. Comparisons use `np.dot()` on unit vectors (mathematically equivalent to cosine). One embed path, one model instance, one normalization step, one distance metric — keeps the silent-failure surface minimal.

- **Multi-vector gallery (multi-photo enrollment).** Each enrolled photo gets its own Qdrant point with a fresh UUID and `payload.identity_id` set to the user-supplied id. `/search` runs `top_k=5`, groups hits by `identity_id`, returns the identity with the highest max cosine across its points. This is the standard production "gallery" pattern — covers pose/lighting/expression variance without changing models.

- **Hosting: two Hugging Face Spaces (one Docker backend + one Streamlit frontend).** HF Spaces gives 16 GB RAM free per Space and supports both Docker and Streamlit SDKs natively. Considered Render free (512 MB too tight for `antelopev2` at ~700 MB resident), Cloud Run (cold-start hurts a demo), Fly.io paid ($2/mo, no cold start), AWS EC2 free tier (t2.micro CPU credits + no HTTPS + manual nginx setup, all wrong shape for a 1-day demo). HF Spaces' single-container-per-Space constraint is why we used **two Spaces** (UI + API) instead of the docker-compose pattern from a Medium reference article — two Spaces communicating over public HTTPS is the HF-idiomatic equivalent.

