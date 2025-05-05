# Use the official PyTorch image with CUDA 12.1 and cuDNN 8 support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# - ffmpeg: For audio/video processing
# - git: Might be needed by some Python packages
# - ca-certificates: Often needed for HTTPS requests by libraries like requests/huggingface_hub
# - iputils-ping: Useful for network debugging inside the container
# --- REMOVED redis-server ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
    iputils-ping && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies (including Flask, Celery, Redis client, Gunicorn)
# Add gunicorn to requirements.txt if using it in entrypoint.sh
# --- ADDED gevent ---
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn gevent # Install gunicorn and gevent here
# --------------------

# --- Install openai-whisper ---
# Install after requirements.txt to leverage Docker layer caching
RUN pip install --no-cache-dir -U openai-whisper
# -----------------------------

# Copy all application code (pipeline modules, web UI, tasks)
COPY *.py /app/

# Copy web UI template and static files
COPY templates /app/templates
COPY static /app/static

# --- REMOVED entrypoint.sh copy and chmod ---
# COPY entrypoint.sh /app/entrypoint.sh
# RUN chmod +x /app/entrypoint.sh
# --------------------------------------------

# Create directories for uploads and outputs (volumes defined in compose will manage these)
# Still useful to ensure they exist within the image structure
RUN mkdir -p /app/audio_input && \
    mkdir -p /app/transcripts_output && \
    echo "Base directories created in image"

# Expose the Flask port (Gunicorn will bind to this)
EXPOSE 5000

# --- REMOVED ENTRYPOINT ---
# ENTRYPOINT ["/app/entrypoint.sh"]
# --------------------------

# --- ADD default CMD (Optional but good practice) ---
# This won't run when compose specifies a command, but helps if running the image directly
CMD ["python", "web_ui.py"]
