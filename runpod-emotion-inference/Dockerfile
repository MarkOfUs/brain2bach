# CPU-friendly base (works on GPU endpoints too, but won't have CUDA-accelerated torch).
FROM python:3.10-slim

WORKDIR /app

# Needed for installing torcheeg from source (and compiling deps like spectrum).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torcheeg v1.1.3 but patch its scipy upper-bound (mirrors the notebook approach).
RUN rm -rf torcheeg_src \
 && git clone --depth 1 --branch v1.1.3 https://github.com/torcheeg/torcheeg.git torcheeg_src \
 && sed -i 's/scipy>=1\.7\.3[[:space:]]*,[[:space:]]*<=\s*1\.10\.1/scipy>=1.7.3/' torcheeg_src/setup.py \
 && pip install --no-cache-dir ./torcheeg_src \
 && rm -rf torcheeg_src

COPY emotion_inference.py handler.py ./

# Put your checkpoint here (see README)
# You must provide this file in ./models before building, or comment this out and use a volume/env-var.
COPY models/cnn_faced_best.pt /app/models/cnn_faced_best.pt

CMD ["python", "-u", "handler.py"]
