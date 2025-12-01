# RunPod SAM Worker Dockerfile
# Use a stable PyTorch image with good numpy compatibility
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use numpy version compatible with PyTorch 2.2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    runpod>=1.6.0 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    git+https://github.com/facebookresearch/segment-anything.git

# Verify numpy-torch compatibility with GPU operations
RUN python -c "import torch; import numpy as np; print(f'Torch: {torch.__version__}, Numpy: {np.__version__}'); t = torch.tensor([1,2,3]); print('CPU:', t.numpy()); print('CUDA available:', torch.cuda.is_available())"

# Download SAM model (vit_b for faster inference)
RUN mkdir -p /app && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O /app/sam_vit_b_01ec64.pth

# Copy handler
COPY handler.py .

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]

