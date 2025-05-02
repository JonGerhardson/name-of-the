# config.py
import os
import torch
import logging

# --- Core Paths & Filenames ---
DEFAULT_OUTPUT_DIR = "/app/transcripts_output"
DEFAULT_LOG_FILENAME = "processing_log.log"
DEFAULT_HF_TOKEN_FILE = "/app/hf-token.txt"
DEFAULT_FAISS_INDEX_FILENAME = "speaker_embeddings.index"
DEFAULT_SPEAKER_MAP_FILENAME = "speaker_names.json"
DEFAULT_IDENTIFIED_JSON_FILENAME = "intermediate_identified_transcript.json"
DEFAULT_FINAL_OUTPUT_FILENAME = "final_normalized_transcript.txt"
EMBEDDING_CACHE_SUBDIR = 'embedding_model_cache' # Cache dir for embedding model

# --- Model Configurations ---
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_DEVICE = "cpu" # Decide default based on common case or force user choice
DEFAULT_WHISPER_COMPUTE = "int8"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PUNCTUATION_MODEL = "felflare/bert-restore-punctuation"

# --- Processing Parameters ---
DEFAULT_PROCESSING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SIMILARITY_THRESHOLD = 0.80
DEFAULT_MIN_SEGMENT_DURATION = 2.0
DEFAULT_PUNCTUATION_CHUNK_SIZE = 256
DEFAULT_EMBEDDING_DIM = 192 # Default embedding dimension if detection fails

# --- Normalization Options ---
DEFAULT_NORMALIZE_NUMBERS = False
DEFAULT_REMOVE_FILLERS = False
DEFAULT_FILLER_WORDS = ['um', 'uh', 'hmm', 'mhm', 'uh huh', 'like', 'you know']

# --- Logging Levels ---
DEFAULT_CONSOLE_LOG_LEVEL = logging.INFO
DEFAULT_FILE_LOG_LEVEL = logging.DEBUG
VERBOSE_CONSOLE_LOG_LEVEL = logging.DEBUG

# --- Argument Parser Defaults ---
# (Can be defined here or directly in the main script's parser setup)

