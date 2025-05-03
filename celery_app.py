# ~/name-of-the/celery_app.py
from celery import Celery
import os
# Import config to potentially use settings, though none are directly used here for broker/backend
import config

# --- Celery Configuration ---

# Define the Redis URL.
# Since Redis is running in the same Docker container (started by entrypoint.sh),
# 'localhost' is the correct hostname. Port 6379 and database 0 are defaults.
REDIS_URL = 'redis://localhost:6379/0'

# Define the Celery application instance.
# - The first argument ('tasks') is the name of the main module where tasks are defined.
#   This is important for Celery's task discovery mechanism.
# - 'broker' specifies the URL for the message broker (Redis in this case).
# - 'backend' specifies the URL for the result backend (also Redis).
# - 'include' is a list of modules to import when the worker starts, ensuring
#   that tasks defined in these modules (@celery_app.task decorators) are registered.
celery = Celery(
    'tasks', # Name of the main module containing tasks
    broker=REDIS_URL,
    backend=REDIS_URL,
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

# Standard boilerplate for potentially running the worker directly (though not used with entrypoint.sh)
if __name__ == '__main__':
    celery.start()

