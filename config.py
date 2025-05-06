# config.py
import os
import torch
import logging
from typing import Optional


# --- Core Paths & Filenames ---
DEFAULT_OUTPUT_DIR = "/app/transcripts_output"
DEFAULT_LOG_FILENAME = "processing_log.log"
DEFAULT_HF_TOKEN_FILE = "/app/hf-token.txt"
DEFAULT_IDENTIFIED_JSON_FILENAME = "intermediate_identified_transcript.json" # Stage 1 output
DEFAULT_FINAL_OUTPUT_FILENAME = "final_transcript.txt" # Standard text output
DEFAULT_FINAL_JSON_FILENAME = "final_transcript.json" # Segment-level final JSON
DEFAULT_SRT_FILENAME = "final_transcript.srt" # SRT format
DEFAULT_TIMESTAMPS_FILENAME = "final_transcript_timestamps.txt" # Text with timestamps
EMBEDDING_CACHE_SUBDIR = 'embedding_model_cache' # Cache dir for embedding model

# --- Central Speaker Database Paths ---
CENTRAL_DB_DIR = os.environ.get('SPEAKER_DB_DIR', "/app/speaker_database")
CENTRAL_FAISS_INDEX_PATH = os.path.join(CENTRAL_DB_DIR, "global_speaker_embeddings.index")
CENTRAL_SPEAKER_MAP_PATH = os.path.join(CENTRAL_DB_DIR, "global_speaker_names.json")

# --- Model Cache Paths ---
WHISPER_CACHE_DIR = os.environ.get('WHISPER_CACHE_DIR', "/app/model_cache/whisper")

# --- Model Configurations ---
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_DEVICE = "cuda"
# DEFAULT_WHISPER_COMPUTE = "int8" # Unused by openai-whisper
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PUNCTUATION_MODEL = "felflare/bert-restore-punctuation"

# --- Processing Parameters ---
DEFAULT_PROCESSING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MIN_SEGMENT_DURATION = 3.0
DEFAULT_PUNCTUATION_CHUNK_SIZE = 256
DEFAULT_EMBEDDING_DIM = 192 # Default embedding dimension if detection fails
DEFAULT_MERGE_GAP_THRESHOLD_MS = 200   # Merge if gap is less than this (milliseconds)


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

# --- Ensure Directories Exist ---
try:
    os.makedirs(CENTRAL_DB_DIR, exist_ok=True)
    print(f"Ensured central speaker database directory exists: {CENTRAL_DB_DIR}")
    os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
    print(f"Ensured Whisper model cache directory exists: {WHISPER_CACHE_DIR}")
except OSError as e:
    print(f"WARNING: Could not create directories {CENTRAL_DB_DIR} or {WHISPER_CACHE_DIR} on import: {e}")


