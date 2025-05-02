# models.py
import os
import torch
import logging
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import pipeline as hf_pipeline # Alias to avoid name conflict
import config # Import your config module

logger = logging.getLogger(__name__)

def load_whisper_model(model_name, device, compute_type):
    """Loads the Faster Whisper model."""
    logger.info(f"Loading Faster Whisper model: {model_name} (Device: {device}, Compute: {compute_type})")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=False)
        logger.info("Faster Whisper model loaded successfully.")
        return model
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading Faster Whisper model '{model_name}': {e}. Cannot proceed.", exc_info=True)
        return None # Indicate failure

def load_diarization_pipeline(model_name, device, hf_token):
    """Loads the Pyannote diarization pipeline."""
    if not hf_token:
        logger.warning("Skipping Pyannote loading (no Hugging Face token provided).")
        return None

    logger.info(f"Loading Pyannote diarization pipeline: {model_name} (Device: {device})")
    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        logger.info(f"Pyannote pipeline loaded successfully to {device}.")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading Pyannote pipeline: {e}. Diarization will be skipped.", exc_info=True)
        return None

def load_embedding_model(model_name, device, cache_dir):
    """Loads the SpeechBrain embedding model."""
    logger.info(f"Loading SpeechBrain embedding model: {model_name} (Device: {device})")
    os.makedirs(cache_dir, exist_ok=True) # Ensure cache dir exists
    try:
        model = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device},
            savedir=cache_dir
        )
        model.eval()
        logger.info("SpeechBrain embedding model loaded successfully.")

        # Attempt to determine embedding dimension
        emb_dim = config.DEFAULT_EMBEDDING_DIM # Default
        try:
            dummy_input = torch.rand(1, 16000).to(torch.device(device)) # 1 sec dummy audio
            with torch.no_grad():
                 output = model.encode_batch(dummy_input)
            emb_dim = output.shape[-1]
            logger.info(f"Determined embedding dimension: {emb_dim}")
        except Exception as emb_e:
            logger.warning(f"Could not dynamically determine embedding dimension: {emb_e}. Using default {emb_dim}.", exc_info=False) # Don't need full trace usually

        return model, emb_dim
    except Exception as e:
        logger.error(f"Error loading SpeechBrain model '{model_name}': {e}. Speaker ID will be skipped.", exc_info=True)
        return None, config.DEFAULT_EMBEDDING_DIM # Return None model, default dim

def load_punctuation_model(model_name, device):
    """Loads the Hugging Face punctuation model."""
    logger.info(f"Loading Punctuation model: {model_name} (Device: {device})")
    try:
        # Map device string to device index expected by transformers pipeline
        # -1 for CPU, 0 for first GPU if 'cuda'
        device_index = 0 if device == "cuda" else -1
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Punctuation model requested CUDA but not available, using CPU (-1).")
            device_index = -1

        punc_pipeline = hf_pipeline(
            "token-classification",
            model=model_name,
            device=device_index,
            aggregation_strategy="simple" # Or another strategy if needed
        )
        logger.info(f"Punctuation model loaded successfully to device index {device_index}.")
        return punc_pipeline
    except Exception as e:
        logger.error(f"Error loading punctuation model '{model_name}': {e}. Punctuation step will be skipped.", exc_info=True)
        return None

