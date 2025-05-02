# pipeline.py (Pass output_dir to Speaker ID - Corrected Call v2)
import os
import time
import json
import argparse
import logging
import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple, List

# Import modules created above
import config
import log_setup
import utils
import persistence
import models
import audio_processing # Ensure this is imported if used directly
import speaker_id
import text_processing

# Import external libraries needed
try:
    from cucco import Cucco
except ImportError:
    Cucco = None
try:
    import inflect
except ImportError:
    inflect = None

# Use a specific logger for this module
logger = logging.getLogger(__name__)

# --- Constants for Temporary Files ---
TEMP_TRANSCRIPT_SUFFIX = "_temp_identified.json"
TEMP_ENROLL_INFO_SUFFIX = "_temp_enrollment_info.json"
SNIPPETS_SUBDIR = "snippets" # Consistent with speaker_id.py
TEXT_SNIPPET_MAX_WORDS = 20 # Max words for text snippet display

# --- Helper for JSON with Numpy ---
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # Handle arrays
            return obj.tolist() # Convert arrays to lists
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def save_enrollment_info(data: List[Dict[str, Any]], filepath: str):
    """Saves the list containing numpy embeddings to JSON."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Successfully saved enrollment info to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save enrollment info to {filepath}: {e}", exc_info=True)
        raise # Re-raise to signal failure

def load_enrollment_info(filepath: str) -> List[Dict[str, Any]]:
    """Loads the list containing embeddings (as lists) from JSON and converts back to numpy."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        processed_data = []
        for item in data:
            if isinstance(item, dict) and 'embedding' in item and isinstance(item['embedding'], list):
                try:
                    item['embedding'] = np.array(item['embedding'], dtype=np.float32)
                    processed_data.append(item)
                except ValueError as ve:
                    logger.warning(f"Could not convert embedding list to numpy array for item {item.get('temp_id', 'N/A')}: {ve}. Skipping item.")
            elif isinstance(item, dict): # Keep items without embeddings or malformed ones
                 processed_data.append(item)
            else:
                 logger.warning(f"Skipping non-dict item found in enrollment info file: {item}")

        logger.info(f"Successfully loaded and processed enrollment info from {filepath}")
        return processed_data
    except FileNotFoundError:
        logger.error(f"Enrollment info file not found: {filepath}")
        raise FileNotFoundError(f"Enrollment info file not found: {filepath}")
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to decode JSON from enrollment info file {filepath}: {jde}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to load or process enrollment info from {filepath}: {e}", exc_info=True)
        raise

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


# ==============================================================
# Helper Functions (Moved to Global Scope)
# ==============================================================
def run_asr(whisper_model, audio_path):
    """Runs the ASR process."""
    logger.info("--- Starting Transcription (ASR) ---")
    if not whisper_model:
        raise ValueError("Whisper model is not available for ASR.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_asr_time = time.time()
    try:
        # Log audio duration before transcription
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            duration_seconds = waveform.shape[1] / sample_rate
            duration_minutes = int(duration_seconds // 60)
            duration_remaining_seconds = duration_seconds % 60
            logger.info(f"Processing audio with duration {duration_minutes:02d}:{duration_remaining_seconds:06.3f}")
        except Exception as audio_err:
            logger.warning(f"Could not determine audio duration before ASR: {audio_err}")

        segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        transcript_results = []
        total_word_count_asr = 0
        for segment in segments:
            words_list = []
            if segment.words:
                try:
                    words_list = [{"word": w.word.strip(), "start": w.start, "end": w.end, "probability": w.probability}
                                  for w in segment.words if w.word is not None and w.start is not None and w.end is not None]
                    total_word_count_asr += len(words_list)
                except AttributeError as ae:
                    logger.warning(f"Attribute error processing words in segment ({segment.start}-{segment.end}): {ae}. Segment words might be incomplete.")
                except Exception as we:
                    logger.warning(f"Unexpected error processing words in segment ({segment.start}-{segment.end}): {we}. Skipping words for this segment.", exc_info=False)

            segment_dict = {
                "start": segment.start, "end": segment.end,
                "text": segment.text, "words": words_list
            }
            transcript_results.append(segment_dict)

        end_asr_time = time.time()
        logger.info(f"Transcription complete. Found {len(transcript_results)} segments, {total_word_count_asr} words. Took {end_asr_time - start_asr_time:.2f} seconds.")
        if not transcript_results: logger.warning("Transcription resulted in empty segments.")
        return transcript_results

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise RuntimeError(f"Transcription failed: {e}") from e

def run_diarization(diarization_pipeline, audio_path):
    """Runs the diarization process."""
    logger.info("--- Starting Diarization ---")
    if not diarization_pipeline:
        logger.warning("Diarization pipeline not available. Skipping diarization.")
        return None

    start_dia_time = time.time()
    try:
        diarization = diarization_pipeline(audio_path)
        diarization_results = [{"start": turn.start, "end": turn.end, "speaker": label}
                               for turn, _, label in diarization.itertracks(yield_label=True)]
        diarization_results.sort(key=lambda x: x['start'])
        end_dia_time = time.time()
        num_turns = len(diarization_results)
        num_speakers = len(set(d['speaker'] for d in diarization_results)) if diarization_results else 0
        logger.info(f"Diarization complete. Found {num_turns} turns, {num_speakers} unique speaker labels. Took {end_dia_time - start_dia_time:.2f} seconds.")
        return diarization_results
    except Exception as e:
        logger.error(f"Error during diarization: {e}. Proceeding without diarization labels.", exc_info=True)
        return None

def combine_asr_diarization(transcript_results, diarization_results):
    """Combines ASR words with speaker labels from diarization."""
    logger.info("--- Combining Transcription and Diarization ---")
    if not transcript_results:
        logger.warning("No transcription results available. Cannot combine.")
        return []

    combined_transcript = []
    total_word_count_combined = 0
    words_assigned_speaker = 0
    start_comb_time = time.time()
    try:
        speaker_map_timeline = []
        if diarization_results:
            diarization_results.sort(key=lambda x: x['start'])
            speaker_map_timeline = [(turn['start'], turn['end'], turn['speaker']) for turn in diarization_results]
            logger.info(f"Using {len(speaker_map_timeline)} sorted diarization segments for speaker assignment.")
        else:
            logger.warning("Diarization results missing or failed. Assigning 'UNKNOWN' speaker to all words.")

        for segment in transcript_results:
            segment_copy = segment.copy()
            words_in_segment = []
            if 'words' in segment and segment['words']:
                total_word_count_combined += len(segment['words'])
                for word_info in segment['words']:
                    word_copy = word_info.copy()
                    assigned_speaker = "UNKNOWN"
                    start = word_info.get('start')
                    end = word_info.get('end')

                    if start is not None and end is not None and speaker_map_timeline:
                        word_midpoint = start + (end - start) / 2
                        found_speaker = False
                        for turn_start, turn_end, speaker in speaker_map_timeline:
                            if turn_start <= word_midpoint < turn_end:
                                assigned_speaker = speaker
                                words_assigned_speaker += 1
                                found_speaker = True
                                break
                        # If no turn explicitly contains the midpoint, assign UNKNOWN
                        # (This handles gaps between diarization turns if any)
                        # if not found_speaker:
                        #     logger.debug(f"Word '{word_info.get('word')}' ({start:.2f}-{end:.2f}) midpoint {word_midpoint:.2f} not in any diarization turn.")
                    elif not speaker_map_timeline:
                         # If no diarization, all speakers are UNKNOWN
                         words_assigned_speaker += 1
                    else:
                        logger.warning(f"Word '{word_info.get('word')}' lacks start/end times. Assigning 'UNKNOWN'.")

                    word_copy['speaker'] = assigned_speaker
                    words_in_segment.append(word_copy)

            segment_copy['words'] = words_in_segment
            combined_transcript.append(segment_copy)

        if speaker_map_timeline:
            logger.info(f"Speaker assignment to words complete. Assigned speakers to {words_assigned_speaker}/{total_word_count_combined} words.")
        else:
            logger.info(f"Assigned 'UNKNOWN' speaker to {words_assigned_speaker}/{total_word_count_combined} words.")
        end_comb_time = time.time()
        logger.info(f"Combination process took {end_comb_time - start_comb_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during transcription/diarization combination: {e}. Combined transcript may be incomplete.", exc_info=True)
        return transcript_results # Fallback

    return combined_transcript


def run_speaker_identification_wrapper(
    combined_transcript: list,
    embedding_model: Any,
    emb_dim: int,
    audio_path: str,
    output_dir: str, # Argument definition is correct here
    processing_device: str,
    min_segment_duration: float,
    similarity_threshold: float,
    faiss_index_path: str,
    speaker_map_path: str
    ) -> Tuple[Optional[list], list]:
    """Wrapper to call the speaker identification logic from speaker_id module."""
    try:
        # Call the actual function in speaker_id.py, passing output_dir
        # **ASSUMPTION:** speaker_id.run_speaker_identification returns a tuple:
        # (updated_transcript, new_speaker_info_list)
        # where new_speaker_info_list contains dicts with 'temp_id', 'original_label', 'start_time', 'end_time', 'embedding'
        return speaker_id.run_speaker_identification(
            combined_transcript=combined_transcript,
            embedding_model=embedding_model,
            emb_dim=emb_dim,
            audio_path=audio_path,
            output_dir=output_dir, # Pass output_dir here
            processing_device=processing_device,
            min_segment_duration=min_segment_duration,
            similarity_threshold=similarity_threshold,
            faiss_index_path=faiss_index_path,
            speaker_map_path=speaker_map_path
        )
    except AttributeError as ae:
         if 'run_speaker_identification' in str(ae):
             logger.critical("Could not find 'run_speaker_identification' function in speaker_id module!")
         else:
             logger.critical(f"AttributeError calling speaker_id.run_speaker_identification: {ae}", exc_info=True)
         raise
    except Exception as e:
        logger.critical(f"Error calling speaker_id.run_speaker_identification: {e}", exc_info=True)
        raise RuntimeError(f"Speaker Identification failed internally: {e}") from e


def run_postprocessing(
    identified_transcript, punctuation_pipeline, punctuation_chunk_size,
    cucco_instance, inflect_engine, normalize_numbers, remove_fillers, filler_words
    ):
    """Runs punctuation and normalization."""
    logger.info("--- Starting Post-Processing (Punctuation & Normalization) ---")
    start_post_time = time.time()

    punctuated_turns = text_processing.apply_punctuation(
        identified_transcript, punctuation_pipeline, punctuation_chunk_size
    )

    if not punctuated_turns:
        logger.warning("Punctuation step returned no turns. Skipping normalization.")
        return []

    normalized_turns = []
    logger.info("Applying text normalization to punctuated turns...")
    num_normalized = 0
    for speaker, text in punctuated_turns:
        try:
             if (normalize_numbers or remove_fillers) and text:
                 normalized_text = text_processing.normalize_text_custom(
                     text, cucco_instance=cucco_instance, inflect_engine=inflect_engine,
                     normalize_numbers=normalize_numbers, remove_fillers=remove_fillers,
                     filler_words=filler_words
                 )
                 normalized_turns.append((speaker, normalized_text))
                 if normalized_text != text: num_normalized += 1
             else:
                 normalized_turns.append((speaker, text))
        except Exception as e:
            logger.error(f"Error normalizing text for speaker {speaker}: {e}. Using punctuated text.", exc_info=True)
            normalized_turns.append((speaker, text)) # Fallback

    if num_normalized > 0: logger.info(f"Normalization applied to {num_normalized} turns.")
    elif normalize_numbers or remove_fillers: logger.info("Normalization enabled but no changes made to punctuated text.")
    else: logger.info("Normalization skipped (options disabled).")

    end_post_time = time.time()
    logger.info(f"Post-processing finished. Took {end_post_time - start_post_time:.2f} seconds.")
    return normalized_turns

# ==============================================================
# Core Pipeline Logic Function (Staged)
# ==============================================================

def run_full_pipeline(
    output_dir: str,
    input_audio: Optional[str] = None,
    enrollment_stage: str = "full",
    log_file: str = config.DEFAULT_LOG_FILENAME,
    output_json_file: str = config.DEFAULT_IDENTIFIED_JSON_FILENAME,
    output_final_file: str = config.DEFAULT_FINAL_OUTPUT_FILENAME,
    faiss_file: str = config.DEFAULT_FAISS_INDEX_FILENAME,
    map_file: str = config.DEFAULT_SPEAKER_MAP_FILENAME,
    hf_token_path: Optional[str] = config.DEFAULT_HF_TOKEN_FILE,
    whisper_model_name: str = config.DEFAULT_WHISPER_MODEL,
    whisper_device_requested: str = config.DEFAULT_WHISPER_DEVICE,
    whisper_compute: str = config.DEFAULT_WHISPER_COMPUTE,
    processing_device_requested: str = config.DEFAULT_PROCESSING_DEVICE,
    embedding_model_name: str = config.DEFAULT_EMBEDDING_MODEL,
    punctuation_model_name: str = config.DEFAULT_PUNCTUATION_MODEL,
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    min_segment_duration: float = config.DEFAULT_MIN_SEGMENT_DURATION,
    punctuation_chunk_size: int = config.DEFAULT_PUNCTUATION_CHUNK_SIZE,
    normalize_numbers: bool = config.DEFAULT_NORMALIZE_NUMBERS,
    remove_fillers: bool = config.DEFAULT_REMOVE_FILLERS,
    filler_words: list = config.DEFAULT_FILLER_WORDS,
    enrollment_map: Optional[Dict[str, str]] = None,
    setup_logging_in_func: bool = False,
    console_log_level: int = config.DEFAULT_CONSOLE_LOG_LEVEL,
    file_log_level: int = config.DEFAULT_FILE_LOG_LEVEL
) -> Dict[str, Any]:
    """
    Executes the transcription pipeline, potentially in stages for web UI enrollment.
    Handles ASR, diarization, speaker ID, enrollment, and post-processing.
    """
    start_total_time = time.time()
    result_payload = {
        "status": "error", "message": "Pipeline not started",
        "log_file_path": None, "db_updated": False
    }

    # --- Basic Validation ---
    if enrollment_stage == "finalize_enrollment" and enrollment_map is None:
         msg = "enrollment_map is required for 'finalize_enrollment' stage."
         logger.error(msg); result_payload["message"] = msg; return result_payload
    if enrollment_stage not in ["full", "identify_only", "finalize_enrollment"]:
        msg = f"Invalid enrollment_stage: {enrollment_stage}"
        logger.error(msg); result_payload["message"] = msg; return result_payload
    if enrollment_stage in ["full", "identify_only"] and not input_audio:
        msg = f"input_audio is required for '{enrollment_stage}' stage."
        logger.error(msg); result_payload["message"] = msg; return result_payload
    if input_audio and not os.path.exists(input_audio):
        msg = f"Input audio file not found: {input_audio}"
        logger.error(msg); result_payload["message"] = msg; return result_payload

    # --- Ensure Output Dir & Construct Paths ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except OSError as e:
        msg = f"CRITICAL: Failed to create output directory '{output_dir}': {e}"
        logger.critical(msg, exc_info=True); result_payload["message"] = msg; return result_payload

    log_file_path = os.path.join(output_dir, log_file)
    faiss_path = os.path.join(output_dir, faiss_file)
    map_path = os.path.join(output_dir, map_file)
    final_json_path = os.path.join(output_dir, output_json_file)
    final_txt_path = os.path.join(output_dir, output_final_file)
    embedding_cache_dir = os.path.join(output_dir, config.EMBEDDING_CACHE_SUBDIR)
    temp_transcript_path = os.path.join(output_dir, "stage1_transcript_temp.json")
    temp_enroll_info_path = os.path.join(output_dir, "stage1_enroll_info_temp.json")
    result_payload["log_file_path"] = log_file_path

    # --- Setup Logging ---
    if setup_logging_in_func:
        log_setup.setup_logging(log_file_path, console_level=console_log_level, file_level=file_log_level)
        logger.info(f"Logging setup performed by run_full_pipeline (Stage: {enrollment_stage}).")
    else:
        logger.info(f"Skipping internal logging setup (Stage: {enrollment_stage}).")

    logger.info("="*30 + f" Stage: {enrollment_stage} " + "="*30)
    if input_audio: logger.info(f"Input Audio: {input_audio}")

    # --- Initialize Resources ---
    hf_token = utils.load_huggingface_token(hf_token_path)
    cucco_instance = Cucco() if Cucco else None
    inflect_engine = inflect.engine() if inflect else None

    # --- Model & State Variables ---
    whisper_model = None
    diarization_pipeline = None
    embedding_model = None
    punctuation_pipeline = None
    uses_cuda = False
    embedding_dim = config.DEFAULT_EMBEDDING_DIM
    identified_transcript = None
    new_speaker_info = [] # This will contain the dicts with temp_id, start_time, end_time etc.
    # Store original kwargs passed to the function if needed later
    original_kwargs_for_result = {
        "normalize_numbers": normalize_numbers,
        "remove_fillers": remove_fillers,
        "punctuation_model_name": punctuation_model_name,
        "punctuation_chunk_size": punctuation_chunk_size,
        "filler_words": filler_words,
        "processing_device_requested": processing_device_requested,
        "embedding_model_name": embedding_model_name
    }


    # --- Stage-Specific Logic ---
    try:
        # ==================================
        # === Stage: identify_only / full ==
        # ==================================
        if enrollment_stage in ["full", "identify_only"]:
            logger.info("--- Running Initial Processing (ASR, Diarization, Identification) ---")

            whisper_device = utils.check_device(whisper_device_requested)
            processing_device = utils.check_device(processing_device_requested)
            uses_cuda = (whisper_device == "cuda") or (processing_device == "cuda")

            logger.info("--- Initializing Models ---")
            start_load_time = time.time()
            whisper_model = models.load_whisper_model(whisper_model_name, whisper_device, whisper_compute)
            if whisper_model is None: raise RuntimeError("Whisper model failed to load.")
            diarization_pipeline = models.load_diarization_pipeline(config.DEFAULT_DIARIZATION_MODEL, processing_device, hf_token)
            embedding_model, embedding_dim = models.load_embedding_model(embedding_model_name, processing_device, embedding_cache_dir)
            end_load_time = time.time()
            logger.info(f"Model loading phase took {end_load_time - start_load_time:.2f} seconds.")

            transcript_results = run_asr(whisper_model, input_audio)
            if transcript_results is None: raise RuntimeError("ASR step failed critically.")

            diarization_results = None
            if diarization_pipeline:
                diarization_results = run_diarization(diarization_pipeline, input_audio)

            combined_transcript = combine_asr_diarization(transcript_results, diarization_results)
            if not combined_transcript:
                logger.warning("Combination step resulted in empty transcript. Using raw ASR results.")
                combined_transcript = transcript_results

            # --- Speaker Identification ---
            # **ASSUMPTION:** run_speaker_identification_wrapper returns (updated_transcript, new_speaker_info_list)
            # where new_speaker_info_list contains dicts with 'temp_id', 'original_label', 'start_time', 'end_time', 'embedding'
            identified_transcript, new_speaker_info = run_speaker_identification_wrapper(
                combined_transcript=combined_transcript,
                embedding_model=embedding_model,
                emb_dim=embedding_dim,
                audio_path=input_audio,
                output_dir=output_dir, # Pass output_dir
                processing_device=processing_device,
                min_segment_duration=min_segment_duration,
                similarity_threshold=similarity_threshold,
                faiss_index_path=faiss_path,
                speaker_map_path=map_path
            )

            # --- Stage-Specific Actions ---
            if enrollment_stage == "identify_only":
                logger.info("--- Identify Only Stage Complete ---")

                # Calculate relative path for original audio
                original_audio_relative_path = None
                try:
                    original_audio_relative_path = os.path.relpath(input_audio, output_dir)
                    if original_audio_relative_path.startswith(".."):
                        logger.warning(f"Calculated relative path '{original_audio_relative_path}' seems outside output_dir '{output_dir}'. Cannot use for playback.")
                        original_audio_relative_path = None # Cannot use this path
                except ValueError:
                    logger.warning(f"Could not determine relative path for {input_audio} within {output_dir}. Cannot use for playback.")

                # Save intermediate transcript (if valid)
                if identified_transcript:
                    persistence.save_intermediate_transcript(identified_transcript, temp_transcript_path)
                else:
                    logger.warning("Identified transcript is empty or None. Cannot save temp transcript.")

                # Prepare speaker info for return (remove embeddings, keep timestamps, add text snippet)
                unknown_speakers_for_return = []
                if new_speaker_info: # Check if list is not empty
                    # Create a map for quick lookup of words by *temp_id*
                    speaker_words_map = defaultdict(list)
                    if identified_transcript: # Use the transcript AFTER speaker ID for correct labels
                        for segment in identified_transcript:
                            if 'words' in segment:
                                for word_info in segment['words']:
                                    speaker_label = word_info.get('speaker') # This is now temp_id (e.g., UNKNOWN_1)
                                    if speaker_label: # Check if speaker label exists
                                        speaker_words_map[speaker_label].append(word_info)
                    # Sort words by start time for each speaker
                    for label in speaker_words_map:
                        speaker_words_map[label].sort(key=lambda x: x.get('start', float('inf')))
                    # --- End building word map ---

                    for speaker_data in new_speaker_info:
                        # Ensure required keys for UI are present
                        if all(k in speaker_data for k in ('temp_id', 'start_time', 'end_time', 'original_label')):
                            temp_id = speaker_data['temp_id']
                            start_time = speaker_data['start_time']
                            end_time = speaker_data['end_time'] # Use the determined end time

                            # --- START: Extract Text Snippet using temp_id ---
                            text_snippet = ""
                            words_in_range = []
                            # **MODIFIED:** Use the temp_id (which is the key in speaker_words_map)
                            if temp_id in speaker_words_map:
                                word_count = 0
                                for word_info in speaker_words_map[temp_id]:
                                    word_start = word_info.get('start')
                                    # Get words starting at or after the speaker's determined start_time
                                    if word_start is not None and word_start >= start_time:
                                         word_text = word_info.get('word', '').strip()
                                         if word_text:
                                             words_in_range.append(word_text)
                                             word_count += 1
                                             if word_count >= TEXT_SNIPPET_MAX_WORDS:
                                                 break # Limit snippet length
                                text_snippet = " ".join(words_in_range)
                                if word_count >= TEXT_SNIPPET_MAX_WORDS:
                                    text_snippet += "..." # Indicate truncation
                            # --- END: Extract Text Snippet ---

                            if not text_snippet: # Handle case where lookup still failed or no words found
                                logger.warning(f"Could not extract text snippet for {temp_id} (orig: {speaker_data['original_label']})")
                                text_snippet = "(No text preview available)"

                            ui_data = {
                                'temp_id': temp_id,
                                'original_label': speaker_data.get('original_label', 'N/A'),
                                'start_time': start_time,
                                'end_time': end_time,
                                'text_snippet': text_snippet # **NEW**
                            }
                            unknown_speakers_for_return.append(ui_data)
                        else:
                             logger.warning(f"Speaker info missing required keys (temp_id, start_time, end_time, original_label): {speaker_data}. Skipping for UI return.")

                    # Save the full info (including embeddings) for programmatic enrollment later
                    # This info does NOT need the text_snippet
                    save_enrollment_info(new_speaker_info, temp_enroll_info_path)
                else:
                    logger.info("No new speakers found, skipping saving of enrollment info file.")

                # Determine final status for this stage
                final_status = "enrollment_required" if unknown_speakers_for_return else "success"
                final_message = (f"Identification complete. Found {len(unknown_speakers_for_return)} unknown speakers."
                                 if unknown_speakers_for_return
                                 else "Identification complete. All speakers known.")

                # Check if relative path is available if enrollment is required
                if final_status == "enrollment_required" and not original_audio_relative_path:
                    logger.error("Enrollment required, but could not determine relative path to original audio. Cannot proceed with UI playback.")
                    result_payload.update({
                        "status": "error",
                        "message": "Enrollment required, but failed to get relative audio path for playback."
                    })
                    # No need to raise exception here, just return the error payload
                else:
                    # Update payload with all necessary info
                    result_payload.update({
                        "status": final_status,
                        "message": final_message,
                        "temp_transcript_path": temp_transcript_path if identified_transcript else None,
                        "temp_enroll_info_path": temp_enroll_info_path if new_speaker_info else None,
                        "unknown_speakers": unknown_speakers_for_return, # List with timestamps & text snippet for UI
                        "output_dir": output_dir,
                        "original_kwargs": original_kwargs_for_result, # Pass collected kwargs
                        "original_audio_path": original_audio_relative_path # **NEW**
                    })
                # Stop here for this stage

            elif enrollment_stage == "full":
                # (Keep existing logic for CLI enrollment and post-processing)
                logger.info("--- Full Pipeline: Handling Enrollment (CLI Mode) ---")
                db_updated_cli = False
                if new_speaker_info:
                    faiss_index_cli = persistence.load_or_create_faiss_index(faiss_path, embedding_dim)
                    speaker_map_cli = persistence.load_or_create_speaker_map(map_path)
                    if faiss_index_cli is not None and speaker_map_cli is not None:
                        db_updated_cli = speaker_id.enroll_new_speakers_cli(
                            new_speaker_info, faiss_index_cli, speaker_map_cli,
                            faiss_path, map_path
                        )
                        result_payload["db_updated"] = db_updated_cli
                    else:
                        logger.error("Failed to load FAISS index/map for CLI enrollment.")

                if identified_transcript:
                    persistence.save_intermediate_transcript(identified_transcript, final_json_path)
                    result_payload["intermediate_json_path"] = final_json_path
                else:
                    logger.warning("Identified transcript is empty or None. Cannot save final intermediate JSON.")

                punctuation_pipeline = models.load_punctuation_model(punctuation_model_name, processing_device)
                final_turns = run_postprocessing(
                    identified_transcript=identified_transcript,
                    punctuation_pipeline=punctuation_pipeline,
                    punctuation_chunk_size=punctuation_chunk_size,
                    cucco_instance=cucco_instance,
                    inflect_engine=inflect_engine,
                    normalize_numbers=normalize_numbers,
                    remove_fillers=remove_fillers,
                    filler_words=filler_words
                )
                persistence.save_final_transcript(final_turns, final_txt_path)
                result_payload["final_transcript_path"] = final_txt_path

                result_payload.update({ "status": "success", "message": "Full pipeline finished successfully." })
                # Full stage complete

        # ==================================
        # === Stage: finalize_enrollment ===
        # ==================================
        elif enrollment_stage == "finalize_enrollment":
            logger.info("--- Running Finalize Enrollment Stage ---")

            try:
                identified_transcript = persistence.load_intermediate_transcript(temp_transcript_path)
                # Load the full speaker info saved previously, including embeddings
                new_speaker_info_with_embeddings = load_enrollment_info(temp_enroll_info_path)
            except FileNotFoundError as fnf_err:
                raise RuntimeError(f"Missing intermediate data file for finalization: {fnf_err}") from fnf_err
            except Exception as load_err:
                raise RuntimeError(f"Failed to load intermediate data for finalization: {load_err}") from load_err

            if not identified_transcript:
                raise RuntimeError(f"Loaded intermediate transcript from {temp_transcript_path} is empty or invalid.")

            # Determine embedding dimension (reloading model is safer)
            temp_processing_device = utils.check_device(processing_device_requested)
            _, temp_emb_dim = models.load_embedding_model(embedding_model_name, temp_processing_device, embedding_cache_dir)
            embedding_dim = temp_emb_dim
            logger.info(f"Determined embedding dimension for DB load: {embedding_dim}")

            faiss_index = persistence.load_or_create_faiss_index(faiss_path, embedding_dim)
            speaker_map = persistence.load_or_create_speaker_map(map_path)

            db_updated_prog = False
            if new_speaker_info_with_embeddings and enrollment_map:
                if faiss_index is None or speaker_map is None:
                    logger.error("Failed to load FAISS index/map for programmatic enrollment. Skipping.")
                else:
                    db_updated_prog = speaker_id.enroll_speakers_programmatic(
                        enrollment_map=enrollment_map,
                        new_speaker_info=new_speaker_info_with_embeddings, # Use the list with embeddings
                        faiss_index=faiss_index,
                        speaker_map=speaker_map,
                        faiss_index_path=faiss_path,
                        speaker_map_path=map_path
                    )
                    result_payload["db_updated"] = db_updated_prog
            else:
                logger.info("No enrollment map provided or no new speakers identified previously. Skipping programmatic enrollment.")

            # Update Transcript with Newly Enrolled Names
            final_speaker_assignments = {}
            all_speakers_in_transcript = set()
            for segment in identified_transcript:
                if 'words' in segment:
                    for word in segment['words']:
                        if 'speaker' in word: all_speakers_in_transcript.add(word['speaker'])

            for speaker_label in all_speakers_in_transcript:
                # Use the name from enrollment_map if available, otherwise keep original label
                final_speaker_assignments[speaker_label] = enrollment_map.get(speaker_label, speaker_label)

            finalized_transcript = speaker_id.update_transcript_speakers(
                identified_transcript, final_speaker_assignments
            )
            persistence.save_intermediate_transcript(finalized_transcript, final_json_path)
            result_payload["intermediate_json_path"] = final_json_path

            # Post-Processing
            processing_device = utils.check_device(processing_device_requested)
            uses_cuda = (processing_device == "cuda") # Update cuda usage flag
            punctuation_pipeline = models.load_punctuation_model(punctuation_model_name, processing_device)
            final_turns = run_postprocessing(
                identified_transcript=finalized_transcript,
                punctuation_pipeline=punctuation_pipeline,
                punctuation_chunk_size=punctuation_chunk_size,
                cucco_instance=cucco_instance,
                inflect_engine=inflect_engine,
                normalize_numbers=normalize_numbers,
                remove_fillers=remove_fillers,
                filler_words=filler_words
            )

            persistence.save_final_transcript(final_turns, final_txt_path)
            result_payload["final_transcript_path"] = final_txt_path

            # Cleanup Temporary Files
            try:
                if os.path.exists(temp_transcript_path): os.remove(temp_transcript_path)
                if os.path.exists(temp_enroll_info_path): os.remove(temp_enroll_info_path)
                # Also remove snippets directory if it exists (might have been created by speaker_id)
                snippet_dir = os.path.join(output_dir, SNIPPETS_SUBDIR)
                if os.path.isdir(snippet_dir):
                    import shutil # Use shutil to remove directory and contents
                    shutil.rmtree(snippet_dir)
                    logger.info(f"Cleaned up snippets directory: {snippet_dir}")
                logger.info("Cleaned up temporary files.")
            except Exception as clean_err:
                logger.warning(f"Failed to clean up temporary files/snippets: {clean_err}")

            result_payload.update({
                "status": "success",
                "message": "Finalization and enrollment complete." if db_updated_prog else "Finalization complete (no new speakers enrolled).",
            })
            # Finalize stage complete

    except Exception as e:
        error_message = f"Pipeline failed during stage '{enrollment_stage}': {e}"
        logger.critical(error_message, exc_info=True)
        result_payload["status"] = "error"
        result_payload["message"] = error_message
        # Ensure cleanup happens even on error
        utils.cleanup_resources(whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline, use_cuda=uses_cuda)
        # Re-raise the exception so Celery task marks as failed
        raise e

    # --- Final Cleanup ---
    logger.info(f"--- Pipeline Stage '{enrollment_stage}' Finishing ---")
    utils.cleanup_resources(whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline,
                            use_cuda=uses_cuda)
    end_total_time = time.time()
    logger.info(f"Total execution time for stage '{enrollment_stage}': {end_total_time - start_total_time:.2f} seconds.")
    logger.info("="*60)

    return result_payload


# ==============================================================
# Command-Line Interface Execution (Updated)
# ==============================================================
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Modular audio transcription and speaker identification pipeline (CLI mode).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments... (keep as before)
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output-dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                        help="Directory to save output files.")
    parser.add_argument("--log-file", type=str, default=config.DEFAULT_LOG_FILENAME)
    parser.add_argument("--output-json-file", type=str, default=config.DEFAULT_IDENTIFIED_JSON_FILENAME)
    parser.add_argument("--output-final-file", type=str, default=config.DEFAULT_FINAL_OUTPUT_FILENAME)
    parser.add_argument("--hf-token", type=str, default=config.DEFAULT_HF_TOKEN_FILE)
    parser.add_argument("--faiss-file", type=str, default=config.DEFAULT_FAISS_INDEX_FILENAME)
    parser.add_argument("--map-file", type=str, default=config.DEFAULT_SPEAKER_MAP_FILENAME)
    parser.add_argument("--whisper-model", type=str, default=config.DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-device", type=str, default=config.DEFAULT_WHISPER_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--whisper-compute", type=str, default=config.DEFAULT_WHISPER_COMPUTE,
                        choices=["int8", "int8_float16", "float16", "float32"])
    parser.add_argument("--processing-device", type=str, default=config.DEFAULT_PROCESSING_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--embedding-model", type=str, default=config.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--punctuation-model", type=str, default=config.DEFAULT_PUNCTUATION_MODEL)
    parser.add_argument("--similarity-threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--min-segment-duration", type=float, default=config.DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--punctuation-chunk-size", type=int, default=config.DEFAULT_PUNCTUATION_CHUNK_SIZE)
    parser.add_argument("--normalize-numbers", action='store_true', default=config.DEFAULT_NORMALIZE_NUMBERS)
    parser.add_argument("--remove-fillers", action='store_true', default=config.DEFAULT_REMOVE_FILLERS)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose console logging (DEBUG level). Default is INFO.")

    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_arguments()
    console_log_level_cli = config.VERBOSE_CONSOLE_LOG_LEVEL if cli_args.verbose else config.DEFAULT_CONSOLE_LOG_LEVEL

    result = run_full_pipeline(
        output_dir=cli_args.output_dir,
        input_audio=cli_args.input_audio,
        enrollment_stage="full",
        log_file=cli_args.log_file,
        output_json_file=cli_args.output_json_file,
        output_final_file=cli_args.output_final_file,
        hf_token_path=cli_args.hf_token,
        faiss_file=cli_args.faiss_file,
        map_file=cli_args.map_file,
        whisper_model_name=cli_args.whisper_model,
        whisper_device_requested=cli_args.whisper_device,
        whisper_compute=cli_args.whisper_compute,
        processing_device_requested=cli_args.processing_device,
        embedding_model_name=cli_args.embedding_model,
        punctuation_model_name=cli_args.punctuation_model,
        similarity_threshold=cli_args.similarity_threshold,
        min_segment_duration=cli_args.min_segment_duration,
        punctuation_chunk_size=cli_args.punctuation_chunk_size,
        normalize_numbers=cli_args.normalize_numbers,
        remove_fillers=cli_args.remove_fillers,
        filler_words=config.DEFAULT_FILLER_WORDS,
        enrollment_map=None,
        setup_logging_in_func=True,
        console_log_level=console_log_level_cli,
        file_log_level=config.DEFAULT_FILE_LOG_LEVEL
    )

    print(f"\nPipeline finished with status: {result['status']}")
    if result['status'] == 'error':
        print(f"Error message: {result['message']}")
        exit(1)
    else:
        if result.get('intermediate_json_path'): print(f"Intermediate JSON: {result['intermediate_json_path']}")
        if result.get('final_transcript_path'): print(f"Final Transcript: {result['final_transcript_path']}")
        print(f"Speaker DB Updated: {result.get('db_updated')}")
        exit(0)


