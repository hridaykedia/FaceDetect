FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs: opencv needs glib + libgl; onnxruntime needs libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake antelopev2 model into the image so first request isn't a 280MB download.
# Step 1 downloads + extracts; the assertion failure is expected (nested-dir bug).
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)" || true
# Step 2 flattens the nested antelopev2/antelopev2/ -> antelopev2/
RUN cd /root/.insightface/models/antelopev2 \
    && if [ -d antelopev2 ]; then mv antelopev2/*.onnx . && rmdir antelopev2; fi
# Step 3 verifies the model loads cleanly now.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(320,320))"

# HF Spaces runs containers as non-root uid 1000. Set up that user
# and give them a copy of the pre-baked model directory.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/.insightface \
    && cp -r /root/.insightface/models /home/user/.insightface/ \
    && chown -R user:user /home/user/.insightface

COPY --chown=user:user app/ ./app/

USER user
ENV HOME=/home/user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
