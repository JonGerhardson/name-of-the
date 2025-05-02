# log_setup.py
import os
import logging

def setup_logging(log_file_path, console_level=logging.INFO, file_level=logging.DEBUG):
    """Configures logging to console and file."""
    # Create directory for log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.DEBUG) # Set root logger to lowest level

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logging setup complete. Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)}, Path: {log_file_path}")

