# ==============================================================================
# CPU-Optimized Dockerfile for Team Voice STT & Speaker ID Pipeline
# ==============================================================================

FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install system dependencies (FFmpeg for audio conversion, libsndfile, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies with extra-index-url for CPU wheels so PyTorch/pyannote resolve cleanly
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application source code
COPY backend /app/backend

# Create storage directories for persistent volume mount
RUN mkdir -p /app/storage/raw \
             /app/storage/processed \
             /app/storage/voice_registry \
             /app/storage/sarvam_outputs

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/profiles || exit 1

# Launch production Uvicorn server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
