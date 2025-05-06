# utils.py
import os
import logging
import torch
import gc

logger = logging.getLogger(__name__)

def check_device(requested_device):
    """Checks GPU availability and returns the validated device string."""
    if requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU.")
            return "cuda"
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        logger.info("Using CPU.")
        return "cpu"

def cleanup_resources(*args, use_cuda=False):
    """Attempts to delete variables and clear CUDA cache."""
    logger.info("Cleaning up resources...")
    deleted_count = 0
    for var in args:
        try:
            del var
            deleted_count += 1
        except NameError:
            pass # Variable might not have been assigned (e.g., due to errors)
        except Exception as e:
            logger.warning(f"Could not delete variable during cleanup: {e}")

    logger.debug(f"Attempted deletion of {deleted_count} variables.")
    gc.collect()
    logger.debug("Garbage collection triggered.")

    if use_cuda and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")

def load_huggingface_token(token_file_path):
    """Loads the Hugging Face token from a file."""
    hf_token = None
    if token_file_path:
        if os.path.exists(token_file_path):
            try:
                with open(token_file_path, 'r') as f:
                    hf_token = f.read().strip()
                if not hf_token:
                    logger.warning(f"HF token file '{token_file_path}' is empty.")
                    hf_token = None # Treat empty file as no token
                else:
                    logger.info("Hugging Face token loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading HF token from {token_file_path}: {e}", exc_info=True)
                hf_token = None
        else:
            logger.warning(f"HF token file not found at '{token_file_path}'.")
    else:
        logger.info("No HF token file specified.")
    return hf_token

