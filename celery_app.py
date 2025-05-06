# ~/name-of-the/celery_app.py
import os
from celery import Celery
# Import config to potentially use settings, though none are directly used here for broker/backend
import config

# --- Celery Configuration ---

# --- Use Environment Variables for Redis URL ---
# Reads the broker and backend URLs from environment variables set by docker-compose.yml.
# Defaults to 'redis://localhost:6379/0' if the environment variables are not set
# (useful for running outside compose, though less relevant now).
redis_broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
redis_backend_url = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
# --- End Environment Variable Usage ---

# Define the Celery application instance.
# - The first argument ('tasks') is the name of the main module where tasks are defined.
#   This is important for Celery's task discovery mechanism.
# - 'broker' specifies the URL for the message broker (Redis in this case, read from env).
# - 'backend' specifies the URL for the result backend (also Redis, read from env).
# - 'include' is a list of modules to import when the worker starts, ensuring
#   that tasks defined in these modules (@celery_app.task decorators) are registered.
celery = Celery(
    'tasks', # Name of the main module containing tasks
    broker=redis_broker_url,  # Use the URL from environment variable
    backend=redis_backend_url, # Use the URL from environment variable
    include=['tasks'] # List of modules to import to find tasks
)

# Optional Celery configuration settings.
celery.conf.update(
    task_serializer='json', # Use JSON for task serialization
    accept_content=['json'],  # Accept JSON content
    result_serializer='json', # Use JSON for result serialization
    timezone='UTC', # Set a default timezone (optional, but good practice)
    enable_utc=True, # Ensure UTC is enabled if using timezone
    # Improve robustness by enabling retry on startup if broker isn't immediately available
    broker_connection_retry_on_startup=True
)

# Standard boilerplate for potentially running the worker directly (though not used with compose commands)
if __name__ == '__main__':
    # This block is less likely to be used directly when running via docker-compose commands
    celery.start()

