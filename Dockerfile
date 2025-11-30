# RunPod SAM Worker Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    git+https://github.com/facebookresearch/segment-anything.git

# Download SAM model (vit_b for faster inference)
RUN mkdir -p /app && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O /app/sam_vit_b_01ec64.pth

# Copy handler
COPY handler.py .

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]

