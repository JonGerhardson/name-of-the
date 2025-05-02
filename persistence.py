# persistence.py
import os
import json
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_or_create_faiss_index(index_path, dimension):
    """Loads a FAISS index or creates a new one."""
    if os.path.exists(index_path):
        logger.info(f"Attempting to load existing FAISS index from {index_path}")
        try:
            index = faiss.read_index(index_path)
            if index.d != dimension:
                logger.warning(f"Index dimension ({index.d}) differs from model dimension ({dimension}). Creating new index.")
                index = faiss.IndexFlatIP(dimension) # Use Inner Product
            else:
                logger.info(f"FAISS index loaded successfully with {index.ntotal} embeddings (Dim: {index.d}).")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}. Creating a new one.", exc_info=True)
            index = faiss.IndexFlatIP(dimension)
    else:
        logger.info(f"FAISS index not found at {index_path}. Creating a new one with dimension {dimension}.")
        index = faiss.IndexFlatIP(dimension)
    return index

def save_faiss_index(index, index_path):
    """Saves the FAISS index to disk."""
    logger.info(f"Saving FAISS index to {index_path} with {index.ntotal} embeddings...")
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        logger.info("FAISS index saved successfully.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}", exc_info=True)

def load_or_create_speaker_map(map_path):
    """Loads the speaker name map (FAISS ID -> Name) from JSON."""
    if os.path.exists(map_path):
        logger.info(f"Loading speaker map from {map_path}")
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                speaker_map = json.load(f)
            # Convert keys back to integers
            speaker_map = {int(k): v for k, v in speaker_map.items()}
            logger.info(f"Speaker map loaded with {len(speaker_map)} entries.")
        except Exception as e:
            logger.error(f"Error loading speaker map: {e}. Creating a new one.", exc_info=True)
            speaker_map = {}
    else:
        logger.info(f"Speaker map not found at {map_path}. Creating a new one.")
        speaker_map = {}
    return speaker_map

def save_speaker_map(speaker_map, map_path):
    """Saves the speaker name map to JSON."""
    logger.info(f"Saving speaker map to {map_path} with {len(speaker_map)} entries...")
    try:
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        # Ensure keys are strings for JSON compatibility
        save_map = {str(k): v for k, v in speaker_map.items()}
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(save_map, f, indent=2)
        logger.info("Speaker map saved successfully.")
    except Exception as e:
        logger.error(f"Error saving speaker map: {e}", exc_info=True)

def save_intermediate_transcript(transcript_data, output_path):
    """Saves the intermediate transcript data to a JSON file."""
    logger.info(f"Saving intermediate transcript to {output_path}...")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use the NumpyEncoder if transcript_data might contain numpy types,
            # although usually it shouldn't at this stage. Standard json.dump is safer.
            json.dump(transcript_data, f, indent=2) # Removed cls=NumpyEncoder unless needed
        logger.info("Intermediate transcript saved successfully.")
    except Exception as e:
        logger.error(f"Error saving intermediate transcript JSON: {e}", exc_info=True)
        # Optionally re-raise if this failure should stop the process
        # raise

# --- START: Added Function ---
def load_intermediate_transcript(input_path):
    """Loads the intermediate transcript data from a JSON file."""
    logger.info(f"Loading intermediate transcript from {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Intermediate transcript file not found: {input_path}")
        raise FileNotFoundError(f"Intermediate transcript file not found: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        logger.info("Intermediate transcript loaded successfully.")
        return transcript_data
    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from intermediate transcript file {input_path}: {jde}", exc_info=True)
        raise # Re-raise JSON errors
    except Exception as e:
        logger.error(f"Error loading intermediate transcript JSON from {input_path}: {e}", exc_info=True)
        raise # Re-raise other unexpected errors
# --- END: Added Function ---


def save_final_transcript(punctuated_turns, output_path):
    """Saves the final, formatted transcript to a text file."""
    logger.info(f"Saving final transcript to {output_path}...")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if punctuated_turns:
                for speaker, text in punctuated_turns:
                    # Final check for valid speaker/text
                    speaker_str = str(speaker) if speaker is not None else "UNKNOWN_SPEAKER"
                    text_str = str(text) if text is not None else ""
                    f.write(f"{speaker_str}:\n{text_str.strip()}\n\n")
                logger.info(f"Final transcript saved successfully ({len(punctuated_turns)} speaker turns).")
            else:
                logger.warning("Final transcript data is empty or None. Saving an empty file.")
                f.write("") # Write empty file if no data
    except Exception as e:
        logger.error(f"Error saving final transcript text file: {e}", exc_info=True)
        # Optionally re-raise
        # raise


