FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface \
    MODEL_CACHE=/workspace/model_cache \
    OUTPUT_DIR=/workspace/output \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.4"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p  ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR}

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt 
    #torch==2.4.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Copy application code
# COPY --chown=appuser:appuser . .
COPY . .
# Expose port
EXPOSE 8000
RUN pwd
RUN ls
RUN pip show transformers
CMD python3.10 -u runpod_handler.py