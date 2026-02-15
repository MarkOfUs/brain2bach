# CPU-friendly base (works on GPU endpoints too, but won't have CUDA-accelerated torch).
FROM python:3.10-slim

WORKDIR /app

# Needed for building some scientific deps, and for libsndfile (soundfile).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch from the PyTorch CPU wheel index (smaller + faster).
# If you want CUDA wheels instead, remove the --index-url line and just pip install torch.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install --no-cache-dir -r requirements.txt

COPY music_inference.py handler.py ./
COPY models/ /app/models/

CMD ["python", "-u", "handler.py"]
