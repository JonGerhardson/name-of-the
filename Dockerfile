# Use the official PyTorch image with CUDA 12.1 and cuDNN 8 support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# - ffmpeg: For audio/video processing
# - git: Might be needed by some Python packages
# - redis-server: For Celery message broker/backend (simple setup)
# - ca-certificates: Often needed for HTTPS requests by libraries like requests/huggingface_hub
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    redis-server \
    ca-certificates && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies (including Flask, Celery, Redis client, Gunicorn)
# Add gunicorn to requirements.txt if using it in entrypoint.sh
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn # Install gunicorn here or add to requirements

# Copy all application code (pipeline modules, web UI, tasks)
COPY *.py /app/

# Copy web UI template and static files
COPY templates /app/templates
COPY static /app/static

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create directories for uploads and outputs (volumes should override these)
# These should ideally exist on the host and be mounted
RUN mkdir -p /app/audio_input && \
    mkdir -p /app/transcripts_output && \
    # Set permissions if needed, e.g., if running container as non-root
    # Example: chown -R 1000:1000 /app/audio_input /app/transcripts_output
    # Using nobody:nogroup might not work if the user doesn't exist or Gunicorn/Flask runs as root
    echo "Directories created"

# Expose the Flask port
EXPOSE 5000

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]

# Note: CMD is not needed when ENTRYPOINT uses exec for the final command

