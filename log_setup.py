# log_setup.py
import os
import logging
# START <<< MODIFICATION >>> Import Optional
from typing import Optional
# END <<< MODIFICATION >>>

# Keep the original setup function if needed for standalone execution (like CLI)
def setup_logging(log_file_path, console_level=logging.INFO, file_level=logging.DEBUG):
    """Configures logging to console and file."""
    # Create directory for log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.DEBUG) # Set root logger to lowest level

    # Prevent duplicate handlers if called multiple times directly
    if logger.hasHandlers():
        logger.handlers.clear()
        # Use print for bootstrap phase logging as handlers might be cleared
        print(f"Cleared existing handlers for root logger in setup_logging.")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s') # Simplified format
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    print(f"Added Console Handler (Level: {logging.getLevelName(console_level)})")


    # File Handler (for standalone/CLI)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    print(f"Added File Handler (Level: {logging.getLevelName(file_level)}, Path: {log_file_path})")


    # Use logging.info only after handlers are added
    logging.info(f"Standard logging setup complete. Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)}")

# --- START: New Helper Functions for Job-Specific Logging ---

def add_job_log_handler(logger_instance: logging.Logger, log_file_path: str, level: int = logging.DEBUG) -> Optional[logging.FileHandler]:
    """
    Adds a FileHandler specifically for a job's log file to the given logger instance.

    Args:
        logger_instance: The logger instance (e.g., logging.getLogger() or a specific logger).
        log_file_path: The full path to the job-specific log file.
        level: The logging level for this file handler.

    Returns:
        The created FileHandler instance, or None if creation failed.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create file handler
        job_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8') # Use 'a' to append if task restarts
        job_handler.setLevel(level)

        # Create formatter (use a detailed one for the file)
        formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        job_handler.setFormatter(formatter)

        # Add handler to the logger
        logger_instance.addHandler(job_handler)
        # Use print for bootstrap phase logging as handlers might not be fully set up
        print(f"Successfully added job log handler for {log_file_path}")
        # Log via the handler itself once added
        logger_instance.info(f"Job-specific file logging enabled to: {log_file_path}")
        return job_handler
    except Exception as e:
        # Log error using the existing handlers (e.g., console)
        logger_instance.error(f"Failed to add job log handler for {log_file_path}: {e}", exc_info=True)
        return None

def remove_job_log_handler(logger_instance: logging.Logger, handler: Optional[logging.FileHandler]):
    """
    Removes a specific FileHandler from the logger instance.

    Args:
        logger_instance: The logger instance.
        handler: The FileHandler instance to remove.
    """
    if handler and isinstance(handler, logging.FileHandler):
        try:
            # Log intention *before* removing the handler
            handler_path = getattr(handler, 'baseFilename', 'N/A') # Get path before closing
            logger_instance.info(f"Removing job log handler for: {handler_path}")
            handler.close() # Close the file stream
            logger_instance.removeHandler(handler)
            # Use print as the handler is now removed from the logger instance
            print(f"Successfully removed job log handler for {handler_path}")
        except Exception as e:
            # Use print as the handler might already be gone or broken
            print(f"ERROR: Exception while removing job log handler for {getattr(handler, 'baseFilename', 'N/A')}: {e}")
    elif handler:
        print(f"WARNING: Attempted to remove a non-FileHandler or invalid handler: {handler}")

# --- END: New Helper Functions ---


