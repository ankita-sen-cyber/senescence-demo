# Full pipeline image (classical + Geneformer).
#
# Build:
#   docker build -t rnaseq-loop .
#
# Run (GPU — requires nvidia-container-toolkit on the host):
#   docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs rnaseq-loop
#
# NOTE for RTX 5090 (Blackwell, sm_120): use a base image with CUDA 12.8+
# and PyTorch 2.6+. Adjust the FROM line to match your host CUDA driver.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU pipeline + GPU/Geneformer)
COPY requirements.txt requirements-gpu.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-gpu.txt

# Geneformer (installed from source — the package lives in the HF repo)
RUN pip install git+https://huggingface.co/ctheodoris/Geneformer

# Application code
COPY . .

# Default command: drop into a shell; run scripts explicitly
CMD ["bash"]
