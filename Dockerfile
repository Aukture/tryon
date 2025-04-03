# # Stage 1: Builder stage
# FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 as builder

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3-opencv \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     unzip \
#     wget \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Create virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Stage 2: Runtime stage
# FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# # Copy virtual environment from builder
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# # MKDIR -p /workspace

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     HF_HOME=/workspace/huggingface \
#     MODEL_CACHE=/workspace/model_cache \
#     OUTPUT_DIR=/workspace/output \
#     TRANSFORMERS_CACHE=/workspace/model_cache \
#     DIFFUSERS_CACHE=/workspace/model_cache


# # Create non-root user
# RUN useradd -m appuser && \
#     mkdir -p ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR} && \
#     chown appuser:appuser ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR}

# USER appuser
# WORKDIR /app

# # Copy application code
# COPY --chown=appuser:appuser . .

# # Expose port
# EXPOSE 8000

# # # Health check
# # HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
# #     CMD curl -f http://localhost:8000/health || exit 1

# # Run the application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Stage 1: Builder stage
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as builder
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

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime
# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt 
    #torch==2.4.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Stage 2: Runtime stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
    python3.10 \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN source /opt/venv/bin/activate 
# Create non-root user
# RUN useradd -m appuser && \
#     mkdir -p ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR} && \
#     chown appuser:appuser ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR}

RUN mkdir -p  -p ${HF_HOME} ${MODEL_CACHE} ${OUTPUT_DIR}

# USER appuser
WORKDIR /app
g
# Copy application code
# COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8000
RUN pwd
RUN ls
# Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# RUN pip install uvicorn
# Run the application
# ENTRYPOINT ["uvicorn", "main:app", "--host" , "0.0.0.0", "--port", "8000"]
CMD python3.10 -u runpod_handler.py