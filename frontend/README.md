---
title: Face Match UI
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Face Match UI

Streamlit frontend for the [Face Match + Vector Search](https://github.com/) API.

Two tabs:
- **Enroll** — upload one or more photos of a person + an identity id; calls `POST /enroll`.
- **Search** — upload a query photo; calls `POST /search` and shows the top match + cosine.

Backend URL is read from the `BACKEND_URL` env var (set in Space Settings → Variables and secrets).
