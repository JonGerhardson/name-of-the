#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting entrypoint script..."

# Set the PyTorch CUDA multiprocessing start method
# Option 1: Recommended modern approach (Tried, didn't work)
# export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# Option 2: Older approach (Trying this now)
export CUDA_START_METHOD=spawn
echo "Set CUDA_START_METHOD=spawn"

# Start Redis server in the background
echo "Starting Redis server..."
redis-server --daemonize yes --loglevel warning
# Wait a moment for Redis to start
sleep 2
echo "Redis server started."

# Start Celery worker in the background
# Point -A to the celery instance in celery_app.py
echo "Starting Celery worker..."
# Consider reducing concurrency (-c) if memory becomes an issue
celery -A celery_app.celery worker --loglevel=info &
# Wait a moment for the worker to initialize
sleep 5
echo "Celery worker started."

# Start the Flask web server
echo "Starting Flask app with Gunicorn..."
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --log-level info web_ui:app

echo "Entrypoint script finished." # This line might not be reached due to exec
