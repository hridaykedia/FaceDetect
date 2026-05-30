"""Streamlit UI for the Face Match + Vector Search API.

Two tabs: Enroll (upload + identity id) and Search (upload only).
Calls the FastAPI service at $BACKEND_URL.
"""
import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
THRESHOLD = 0.40  # mirrors the backend default

st.set_page_config(page_title="Face Match", page_icon="🔍", layout="centered")
st.title("🔍 Face Match + Vector Search")

if not BACKEND_URL:
    st.error(
        "`BACKEND_URL` is not set. In **Space settings → Variables and secrets**, "
        "set `BACKEND_URL` to the FastAPI Space's public URL "
        "(e.g. `https://<user>-face-match-api.hf.space`)."
    )
    st.stop()

st.caption(f"Backend: `{BACKEND_URL}` · [API docs]({BACKEND_URL}/docs)")

tab_enroll, tab_search = st.tabs(["📥 Enroll", "🔎 Search"])

with tab_enroll:
    st.subheader("Enroll a face")
    identity = st.text_input("Identity id", placeholder="e.g. alice")
    photos = st.file_uploader(
        "Photo(s) of this person — multiple uploads enroll under the same id",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Enroll", disabled=not (identity and photos), key="enroll_btn"):
        with st.spinner(f"Enrolling {len(photos)} photo(s) for `{identity}` ..."):
            files = [("image", (p.name, p.getvalue(), p.type or "image/jpeg")) for p in photos]
            try:
                r = httpx.post(
                    f"{BACKEND_URL}/enroll",
                    params={"id": identity},
                    files=files,
                    timeout=120.0,
                )
                r.raise_for_status()
                data = r.json()
                st.success(f"Stored {data['stored']} photo(s) under id `{data['id']}`")
                with st.expander("Raw response"):
                    st.json(data)
            except httpx.HTTPStatusError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except httpx.HTTPError as e:
                st.error(f"Request failed: {e}")

with tab_search:
    st.subheader("Search for a match")
    query = st.file_uploader(
        "Query photo",
        type=["jpg", "jpeg", "png"],
        key="search_upload",
    )
    if query is not None:
        st.image(query, width=240, caption="Query")

    if st.button("Search", disabled=query is None, key="search_btn"):
        with st.spinner("Searching ..."):
            files = {"image": (query.name, query.getvalue(), query.type or "image/jpeg")}
            try:
                r = httpx.post(f"{BACKEND_URL}/search", files=files, timeout=120.0)
                r.raise_for_status()
                data = r.json()

                col_match, col_lat = st.columns(2)
                col_match.metric(
                    "Top match",
                    data["match_id"] or "—",
                    delta=f"cosine {data['cosine']:.4f}" if data["cosine"] is not None else "",
                    delta_color="normal",
                )
                col_lat.metric("Qdrant ANN latency", f"{data['latency_ms']} ms")

                if data["above_threshold"]:
                    st.success(f"✅ Match accepted — cosine ≥ {THRESHOLD} threshold")
                else:
                    st.warning(f"⚠️ Below {THRESHOLD} threshold — not a confident match")

                with st.expander("Raw response"):
                    st.json(data)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    st.error("No face detected in the uploaded image.")
                else:
                    st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except httpx.HTTPError as e:
                st.error(f"Request failed: {e}")

st.divider()
st.caption("InsightFace `antelopev2` (SCRFD-10G + glintr100 ArcFace) → Qdrant Cloud HNSW.")
