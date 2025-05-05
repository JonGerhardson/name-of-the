# config.py
import os
import torch
import logging

# --- Core Paths & Filenames ---
DEFAULT_OUTPUT_DIR = "/app/transcripts_output"
DEFAULT_LOG_FILENAME = "processing_log.log"
DEFAULT_HF_TOKEN_FILE = "/app/hf-token.txt"
# Job-specific filenames (remain relative to job output dir)
DEFAULT_IDENTIFIED_JSON_FILENAME = "intermediate_identified_transcript.json"
DEFAULT_FINAL_OUTPUT_FILENAME = "final_normalized_transcript.txt"
EMBEDDING_CACHE_SUBDIR = 'embedding_model_cache' # Cache dir for embedding model

# --- Central Speaker Database Paths --- START <<< MODIFICATION >>>
# Define a persistent directory accessible by all workers/web server
# This path should ideally be mapped to a Docker volume or persistent host directory
CENTRAL_DB_DIR = os.environ.get('SPEAKER_DB_DIR', "/app/speaker_database") # Use env var or default
CENTRAL_FAISS_INDEX_PATH = os.path.join(CENTRAL_DB_DIR, "global_speaker_embeddings.index")
CENTRAL_SPEAKER_MAP_PATH = os.path.join(CENTRAL_DB_DIR, "global_speaker_names.json")
# --- Central Speaker Database Paths --- END <<< MODIFICATION >>>

# --- Model Configurations ---
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_DEVICE = "cuda" # Changed based on previous step
DEFAULT_WHISPER_COMPUTE = "int8" # Note: This is unused by openai-whisper
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PUNCTUATION_MODEL = "felflare/bert-restore-punctuation"

# --- Processing Parameters ---
DEFAULT_PROCESSING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MIN_SEGMENT_DURATION = 3.0
DEFAULT_PUNCTUATION_CHUNK_SIZE = 256
DEFAULT_EMBEDDING_DIM = 192 # Default embedding dimension if detection fails

# --- Normalization Options ---
DEFAULT_NORMALIZE_NUMBERS = False # Corrected typo from FalseI
DEFAULT_REMOVE_FILLERS = False
DEFAULT_FILLER_WORDS = ['um', 'uh', 'hmm', 'mhm', 'uh huh', 'like', 'you know']

# --- Logging Levels ---
DEFAULT_CONSOLE_LOG_LEVEL = logging.INFO
DEFAULT_FILE_LOG_LEVEL = logging.DEBUG
VERBOSE_CONSOLE_LOG_LEVEL = logging.DEBUG

# --- Argument Parser Defaults ---
# (Can be defined here or directly in the main script's parser setup)

# --- START <<< MODIFICATION >>> ---
# Ensure the central database directory exists at startup if possible
# This helps prevent errors if the volume/mount isn't ready immediately
try:
    os.makedirs(CENTRAL_DB_DIR, exist_ok=True)
    logging.info(f"Ensured central speaker database directory exists: {CENTRAL_DB_DIR}")
except OSError as e:
    logging.error(f"Could not create central speaker database directory {CENTRAL_DB_DIR}: {e}")
# --- END <<< MODIFICATION >>> ---
