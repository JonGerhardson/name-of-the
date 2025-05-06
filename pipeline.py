# pipeline.py (Pass output_dir to Speaker ID - Corrected Call v2)
# Updated to use openai-whisper
# Includes explicit Whisper transcription parameters in run_asr
# Uses Central Speaker Database Paths
# Integrated whisper_prompt functionality
# Integrated whisper_language functionality
# REFINED: Added Pyannote kwargs passing and controllable post-ID merge logic
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
# import log_setup # Logging setup handled by caller (tasks.py or CLI main)
import utils
import persistence
import models # Uses the updated models.py
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

# ... (save_enrollment_info, load_enrollment_info, load_intermediate_transcript remain the same) ...
def save_enrollment_info(data: List[Dict[str, Any]], filepath: str):
    """Saves the list containing numpy embeddings to JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Successfully saved enrollment info to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save enrollment info to {filepath}: {e}", exc_info=True)
        raise

def load_enrollment_info(filepath: str) -> List[Dict[str, Any]]:
    """Loads the list containing embeddings (as lists) from JSON and converts back to numpy."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_data = []
        for item in data:
            if isinstance(item, dict) and 'embedding' in item and isinstance(item['embedding'], list):
                try:
                    item['embedding'] = np.array(item['embedding'], dtype=np.float32)
                    processed_data.append(item)
                except ValueError as ve:
                    logger.warning(f"Could not convert embedding list to numpy array for item {item.get('temp_id', 'N/A')}: {ve}. Skipping item.")
            elif isinstance(item, dict):
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
        raise
    except Exception as e:
        logger.error(f"Error loading intermediate transcript JSON from {input_path}: {e}", exc_info=True)
        raise


# ==============================================================
# Helper Functions (Moved to Global Scope)
# ==============================================================
def run_asr(whisper_model, audio_path, whisper_prompt: str = "", whisper_language: Optional[str] = None):
    """Runs the ASR process using openai-whisper."""
    logger.info("--- Starting Transcription (ASR with openai-whisper) ---")
    if not whisper_model:
        raise ValueError("Whisper model is not available for ASR.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_asr_time = time.time()
    try:
        # Log audio duration
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            duration_seconds = waveform.shape[1] / sample_rate
            duration_minutes = int(duration_seconds // 60)
            duration_remaining_seconds = duration_seconds % 60
            logger.info(f"Processing audio with duration {duration_minutes:02d}:{duration_remaining_seconds:06.3f}")
        except Exception as audio_err:
            logger.warning(f"Could not determine audio duration before ASR: {audio_err}")

        # Define transcription options
        transcribe_options = dict(
            word_timestamps=True,
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False
        )
        if whisper_language:
            transcribe_options["language"] = whisper_language
            logger.info(f"Using Whisper language: '{whisper_language}'")
        else:
            logger.info("Using Whisper language auto-detection.")
        if whisper_prompt:
            transcribe_options["prompt"] = whisper_prompt
            logger.info(f"Using Whisper prompt: '{whisper_prompt[:100]}...'")

        logger.info(f"Calling whisper_model.transcribe with options: {transcribe_options}")
        result = whisper_model.transcribe(audio_path, **transcribe_options)

        if 'language' in result:
             logger.info(f"Whisper detected language: '{result['language']}'")
        else:
             logger.warning("Whisper result dictionary does not contain 'language' key.")

        # Process results
        transcript_results = []
        total_word_count_asr = 0
        if 'segments' in result:
            for segment in result['segments']:
                words_list = []
                if 'words' in segment:
                    try:
                        words_list = [{"word": w['word'].strip(),
                                       "start": w['start'],
                                       "end": w['end'],
                                       "probability": w.get('probability', None)}
                                      for w in segment['words'] if 'word' in w and 'start' in w and 'end' in w]
                        total_word_count_asr += len(words_list)
                    except KeyError as ke:
                         logger.warning(f"Key error processing words in segment ({segment.get('start', '?')}-{segment.get('end', '?')}): {ke}. Segment words might be incomplete.")
                    except Exception as we:
                        logger.warning(f"Unexpected error processing words in segment ({segment.get('start', '?')}-{segment.get('end', '?')}): {we}. Skipping words for this segment.", exc_info=False)
                segment_dict = {
                    "start": segment.get('start'), "end": segment.get('end'),
                    "text": segment.get('text', ''), "words": words_list
                }
                transcript_results.append(segment_dict)
        else:
            logger.warning("Transcription result dictionary does not contain 'segments' key.")

        end_asr_time = time.time()
        logger.info(f"Transcription complete. Found {len(transcript_results)} segments, {total_word_count_asr} words. Took {end_asr_time - start_asr_time:.2f} seconds.")
        if not transcript_results: logger.warning("Transcription resulted in empty segments.")
        return transcript_results

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise RuntimeError(f"Transcription failed: {e}") from e

def run_diarization(diarization_pipeline, audio_path, **diarization_kwargs):
    """Runs the diarization process, optionally passing kwargs to the pipeline."""
    logger.info("--- Starting Diarization ---")
    if not diarization_pipeline:
        logger.warning("Diarization pipeline not available. Skipping diarization.")
        return None

    start_dia_time = time.time()
    try:
        # Attempt to pass kwargs directly to the pipeline call
        # IMPORTANT: Consult pyannote.audio docs for valid parameter names for the specific pipeline
        if diarization_kwargs:
            logger.info(f"Attempting to pass keyword arguments to diarization pipeline: {diarization_kwargs}")
            # Use try-except to catch potential errors from invalid kwargs
            try:
                # You might need to instantiate the pipeline with parameters instead:
                # pipeline = Pipeline.from_pretrained(config.DEFAULT_DIARIZATION_MODEL, ...)
                # pipeline.instantiate({"segmentation": {"threshold": diarization_kwargs.get('segmentation_threshold', 0.5)}, ...})
                # Then call: diarization = pipeline(audio_path)
                # For now, we try direct passing:
                diarization = diarization_pipeline(audio_path, **diarization_kwargs)
            except TypeError as te:
                 logger.error(f"TypeError passing kwargs to diarization pipeline: {te}. "
                              f"Check if kwargs {list(diarization_kwargs.keys())} are valid for the pipeline's __call__ method "
                              f"or if they need to be set via configuration/instantiate. "
                              f"Falling back to default diarization.", exc_info=False)
                 # Fallback to default call if direct kwargs are not accepted
                 diarization = diarization_pipeline(audio_path)
            except Exception as pipe_e:
                 logger.error(f"Unexpected error during diarization pipeline call with kwargs: {pipe_e}. "
                              f"Falling back to default diarization.", exc_info=True)
                 # Fallback to default call on other errors
                 diarization = diarization_pipeline(audio_path)
        else:
            logger.info("Running diarization pipeline with default parameters.")
            diarization = diarization_pipeline(audio_path)

        if diarization is None:
             logger.error("Diarization pipeline returned None. Cannot proceed with diarization results.")
             return None

        # Process the results
        diarization_results = [{"start": turn.start, "end": turn.end, "speaker": label}
                               for turn, _, label in diarization.itertracks(yield_label=True)]
        diarization_results.sort(key=lambda x: x['start'])
        end_dia_time = time.time()
        num_turns = len(diarization_results)
        num_speakers = len(set(d['speaker'] for d in diarization_results)) if diarization_results else 0
        logger.info(f"Diarization complete. Found {num_turns} turns, {num_speakers} unique speaker labels. Took {end_dia_time - start_dia_time:.2f} seconds.")
        return diarization_results
    except Exception as e:
        # Catch errors during result processing as well
        logger.error(f"Error during diarization processing: {e}. Proceeding without diarization labels.", exc_info=True)
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
                        # Find the turn containing the word midpoint
                        for turn_start, turn_end, speaker in speaker_map_timeline:
                            # Use strict inequality for end time as turns are often [start, end)
                            if turn_start <= word_midpoint < turn_end:
                                assigned_speaker = speaker
                                words_assigned_speaker += 1
                                found_speaker = True
                                break
                    elif not speaker_map_timeline:
                         words_assigned_speaker += 1 # All unknown if no diarization
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
    output_dir: str, # Job output dir (for snippets)
    processing_device: str,
    min_segment_duration: float,
    similarity_threshold: float,
    faiss_index_path: str, # Now expecting CENTRAL path
    speaker_map_path: str  # Now expecting CENTRAL path
    ) -> Tuple[Optional[list], list]:
    """Wrapper to call the speaker identification logic from speaker_id module."""
    try:
        return speaker_id.run_speaker_identification(
            combined_transcript=combined_transcript,
            embedding_model=embedding_model,
            emb_dim=emb_dim,
            audio_path=audio_path,
            output_dir=output_dir,
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

def merge_short_speaker_segments(transcript_data: list, time_threshold_sec: float) -> list:
    """
    Merges consecutive word segments assigned to different speakers if the time gap is below a threshold.
    Operates in-place on the 'words' lists within the transcript_data structure.
    """
    logger.info(f"--- Applying post-ID speaker segment merging (Threshold: {time_threshold_sec}s) ---")
    if not transcript_data or time_threshold_sec < 0:
        logger.warning("Skipping segment merging: Invalid input data or threshold.")
        return transcript_data

    merged_count = 0
    for segment_idx, segment in enumerate(transcript_data):
        if 'words' not in segment or not isinstance(segment['words'], list) or len(segment['words']) < 2:
            continue

        words = segment['words']
        i = 0
        while i < len(words) - 1:
            current_word = words[i]
            next_word = words[i+1]

            current_speaker = current_word.get('speaker')
            next_speaker = next_word.get('speaker')
            end_time = current_word.get('end')
            start_time = next_word.get('start')

            if (current_speaker is not None and next_speaker is not None and
                current_speaker != next_speaker and
                end_time is not None and start_time is not None):

                if (start_time >= end_time and
                    (start_time - end_time) < time_threshold_sec):

                    logger.debug(f"Merging speaker '{next_speaker}' into '{current_speaker}' at segment {segment_idx}, word index {i+1} "
                                 f"(Gap: {start_time - end_time:.3f}s < {time_threshold_sec}s)")

                    j = i + 1
                    while j < len(words) and words[j].get('speaker') == next_speaker:
                        words[j]['speaker'] = current_speaker # Relabel IN-PLACE
                        merged_count += 1
                        j += 1
                    i = j
                    continue
                else:
                    i += 1
            else:
                i += 1

    if merged_count > 0:
        logger.info(f"Post-ID merging complete. Relabeled {merged_count} words across short gaps.")
    else:
        logger.info("Post-ID merging complete. No segments met the merging criteria.")

    return transcript_data

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
    output_formats: List[str] = ['txt'],
    hf_token_path: Optional[str] = config.DEFAULT_HF_TOKEN_FILE,
    whisper_model_name: str = config.DEFAULT_WHISPER_MODEL,
    whisper_device_requested: str = config.DEFAULT_WHISPER_DEVICE,
    whisper_cache_path: str = config.WHISPER_CACHE_DIR,
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
    whisper_prompt: str = "",
    whisper_language: Optional[str] = None,
    diarization_kwargs: Optional[Dict[str, Any]] = None,
    enable_merge_heuristic: bool = False,
    merge_gap_threshold_ms: int = config.DEFAULT_MERGE_GAP_THRESHOLD_MS # <<< Changed
) -> Dict[str, Any]:
    """
    Executes the transcription pipeline, potentially in stages for web UI enrollment.
    Handles ASR, diarization, speaker ID, enrollment, and post-processing.
    Outputs transcripts in specified formats.
    """
    start_total_time = time.time()
    log_file_path = os.path.join(output_dir, log_file)
    result_payload = {
        "status": "error", "message": "Pipeline not started",
        "log_file_path": log_file_path,
        "db_updated": False,
        "output_files": {}
    }

    # ... (Validation, Path Construction, Logging Setup, Resource Init remain the same) ...
    # Ensure paths for all potential output files are constructed
    output_path_standard_txt = os.path.join(output_dir, config.DEFAULT_FINAL_OUTPUT_FILENAME)
    output_path_final_json = os.path.join(output_dir, config.DEFAULT_FINAL_JSON_FILENAME) # Path for segment-level JSON
    output_path_srt = os.path.join(output_dir, config.DEFAULT_SRT_FILENAME)
    output_path_timestamps_txt = os.path.join(output_dir, config.DEFAULT_TIMESTAMPS_FILENAME)
    # Path for the *intermediate* word-level JSON (saved in stage 1)
    temp_transcript_path = os.path.join(output_dir, config.DEFAULT_IDENTIFIED_JSON_FILENAME) # Use the constant name
    temp_enroll_info_path = os.path.join(output_dir, "stage1_enroll_info_temp.json") # Job specific temp file
    embedding_cache_dir = os.path.join(output_dir, config.EMBEDDING_CACHE_SUBDIR)

    central_faiss_path = config.CENTRAL_FAISS_INDEX_PATH
    central_map_path = config.CENTRAL_SPEAKER_MAP_PATH
    # ... logging initial settings ...
    logger.info("="*30 + f" Stage: {enrollment_stage} " + "="*30)
    if input_audio: logger.info(f"Input Audio: {input_audio}")
    # ... log other settings ...
    logger.info(f"Requested output formats: {output_formats}")

    hf_token = utils.load_huggingface_token(hf_token_path)
    cucco_instance = Cucco() if Cucco else None
    inflect_engine = inflect.engine() if inflect else None

    # ... (Model & State Variables init) ...
    whisper_model = None
    diarization_pipeline = None
    embedding_model = None
    punctuation_pipeline = None
    uses_cuda = False
    embedding_dim = config.DEFAULT_EMBEDDING_DIM
    identified_transcript = None # After ID + Merge
    finalized_transcript = None # After enrollment name update
    new_speaker_info = []
    original_kwargs_for_result = { # Store settings used
        # ... other args ...
        "output_formats": output_formats
    }


    try:
        # ==================================
        # === Stage: identify_only / full ==
        # ==================================
        if enrollment_stage in ["full", "identify_only"]:
            # ... (ASR, Diarization, Combine, Speaker ID, Merge steps remain the same) ...
            logger.info("--- Running Initial Processing (ASR, Diarization, Identification) ---")

            whisper_device = utils.check_device(whisper_device_requested)
            processing_device = utils.check_device(processing_device_requested)
            uses_cuda = (whisper_device == "cuda") or (processing_device == "cuda")

            logger.info("--- Initializing Models ---")
            start_load_time = time.time()
            whisper_model = models.load_whisper_model(
                whisper_model_name, whisper_device, download_root=whisper_cache_path
            )
            if whisper_model is None: raise RuntimeError("Whisper model failed to load.")
            diarization_pipeline = models.load_diarization_pipeline(config.DEFAULT_DIARIZATION_MODEL, processing_device, hf_token)
            embedding_model, embedding_dim = models.load_embedding_model(embedding_model_name, processing_device, embedding_cache_dir)
            end_load_time = time.time()
            logger.info(f"Model loading phase took {end_load_time - start_load_time:.2f} seconds.")

            transcript_results = run_asr(
                whisper_model, input_audio,
                whisper_prompt=whisper_prompt, whisper_language=whisper_language
            )
            if transcript_results is None: raise RuntimeError("ASR step failed critically.")

            diarization_results = None
            if diarization_pipeline:
                effective_diarization_kwargs = diarization_kwargs if diarization_kwargs is not None else {}
                diarization_results = run_diarization(diarization_pipeline, input_audio, **effective_diarization_kwargs)

            combined_transcript = combine_asr_diarization(transcript_results, diarization_results)
            if not combined_transcript:
                logger.warning("Combination step resulted in empty transcript. Using raw ASR results.")
                combined_transcript = transcript_results

            logger.info(f"Running speaker identification using DB: {central_faiss_path}, {central_map_path}")
            identified_transcript, new_speaker_info = run_speaker_identification_wrapper(
                combined_transcript=combined_transcript,
                embedding_model=embedding_model,
                emb_dim=embedding_dim,
                audio_path=input_audio,
                output_dir=output_dir,
                processing_device=processing_device,
                min_segment_duration=min_segment_duration,
                similarity_threshold=similarity_threshold,
                faiss_index_path=central_faiss_path,
                speaker_map_path=central_map_path
            )

            if enable_merge_heuristic and identified_transcript:
                 merge_threshold_sec = merge_gap_threshold_ms / 1000.0
                 identified_transcript = merge_short_speaker_segments(identified_transcript, time_threshold_sec=merge_threshold_sec)


            # --- Stage-Specific Actions ---
            if enrollment_stage == "identify_only":
                 logger.info("--- Identify Only Stage Complete ---")
                 # Save the potentially merged, word-level transcript as the intermediate state
                 if identified_transcript:
                     # Use the renamed function for clarity
                     persistence.save_word_level_transcript(identified_transcript, temp_transcript_path)
                 else:
                     logger.warning("Identified transcript is empty or None after potential merge. Cannot save temp transcript.")
                 # ... (rest of identify_only logic to prepare UI data) ...
                 original_audio_relative_path = None
                 try:
                     original_audio_relative_path = os.path.relpath(input_audio, output_dir)
                     if original_audio_relative_path.startswith(".."):
                         logger.warning(f"Calculated relative path '{original_audio_relative_path}' seems outside output_dir '{output_dir}'. Cannot use for playback.")
                         original_audio_relative_path = None
                 except ValueError:
                     logger.warning(f"Could not determine relative path for {input_audio} within {output_dir}. Cannot use for playback.")

                 unknown_speakers_for_return = []
                 if new_speaker_info:
                     # ... (logic to build unknown_speakers_for_return with snippets) ...
                     speaker_words_map = defaultdict(list)
                     if identified_transcript: # Use the transcript AFTER speaker ID and merge for correct labels
                         for segment in identified_transcript:
                             if 'words' in segment:
                                 for word_info in segment['words']:
                                     speaker_label = word_info.get('speaker') # This is now temp_id (e.g., UNKNOWN_1) or known name
                                     if speaker_label: # Check if speaker label exists
                                          speaker_words_map[speaker_label].append(word_info)
                     for label in speaker_words_map:
                         speaker_words_map[label].sort(key=lambda x: x.get('start', float('inf')))

                     for speaker_data in new_speaker_info: # new_speaker_info still contains original UNKNOWN labels
                         if all(k in speaker_data for k in ('temp_id', 'start_time', 'end_time', 'original_label')):
                             temp_id = speaker_data['temp_id']
                             start_time = speaker_data['start_time']
                             end_time = speaker_data['end_time']
                             text_snippet = ""
                             words_in_range = []
                             # Look up words using the temp_id from new_speaker_info in the potentially merged map
                             if temp_id in speaker_words_map:
                                 word_count = 0
                                 for word_info in speaker_words_map[temp_id]:
                                     word_start = word_info.get('start')
                                     if word_start is not None and word_start >= start_time:
                                          word_text = word_info.get('word', '').strip()
                                          if word_text:
                                              words_in_range.append(word_text)
                                              word_count += 1
                                              if word_count >= TEXT_SNIPPET_MAX_WORDS:
                                                  break
                                 text_snippet = " ".join(words_in_range)
                                 if word_count >= TEXT_SNIPPET_MAX_WORDS:
                                     text_snippet += "..."
                             if not text_snippet:
                                 logger.warning(f"Could not extract text snippet for {temp_id} (orig: {speaker_data['original_label']})")
                                 text_snippet = "(No text preview available)"

                             ui_data = {
                                 'temp_id': temp_id,
                                 'original_label': speaker_data.get('original_label', 'N/A'),
                                 'start_time': start_time,
                                 'end_time': end_time,
                                 'text_snippet': text_snippet
                             }
                             unknown_speakers_for_return.append(ui_data)
                         else:
                              logger.warning(f"Speaker info missing required keys (temp_id, start_time, end_time, original_label): {speaker_data}. Skipping for UI return.")

                     save_enrollment_info(new_speaker_info, temp_enroll_info_path)
                 else:
                     logger.info("No new speakers found, skipping saving of enrollment info file.")

                 final_status = "enrollment_required" if unknown_speakers_for_return else "success"
                 final_message = (f"Identification complete. Found {len(unknown_speakers_for_return)} unknown speakers."
                                  if unknown_speakers_for_return
                                  else "Identification complete. All speakers known.")

                 if final_status == "enrollment_required" and not original_audio_relative_path:
                     logger.error("Enrollment required, but could not determine relative path to original audio. Cannot proceed with UI playback.")
                     result_payload.update({
                         "status": "error",
                         "message": "Enrollment required, but failed to get relative audio path for playback."
                     })
                 else:
                     result_payload.update({
                         "status": final_status,
                         "message": final_message,
                         "temp_transcript_path": temp_transcript_path if identified_transcript else None, # Use renamed path var
                         "temp_enroll_info_path": temp_enroll_info_path if new_speaker_info else None,
                         "unknown_speakers": unknown_speakers_for_return,
                         "output_dir": output_dir,
                         "original_kwargs": original_kwargs_for_result, # Include all args used
                         "original_audio_path": original_audio_relative_path
                     })
                 # Stop here for identify_only stage

            elif enrollment_stage == "full":
                 logger.info("--- Full Pipeline: Handling Enrollment (CLI Mode) & Final Output ---")
                 db_updated_cli = False
                 if new_speaker_info:
                     # ... (CLI enrollment logic remains the same) ...
                     logger.info(f"Loading CENTRAL speaker DB for CLI enrollment: {central_faiss_path}, {central_map_path}")
                     faiss_index_cli = persistence.load_or_create_faiss_index(central_faiss_path, embedding_dim)
                     speaker_map_cli = persistence.load_or_create_speaker_map(central_map_path)
                     if faiss_index_cli is not None and speaker_map_cli is not None:
                         db_updated_cli = speaker_id.enroll_new_speakers_cli(
                             new_speaker_info, faiss_index_cli, speaker_map_cli,
                             central_faiss_path, central_map_path
                         )
                         result_payload["db_updated"] = db_updated_cli
                     else:
                         logger.error("Failed to load CENTRAL FAISS index/map for CLI enrollment.")


                 # Use the identified (and potentially merged) transcript for final outputs
                 finalized_transcript = identified_transcript # In full mode, this is the final structure
                 if not finalized_transcript:
                      logger.error("Cannot proceed to final output generation: finalized_transcript is missing.")
                      raise RuntimeError("Transcript data missing before final output generation.")

                 # START <<< MODIFICATION >>> Generate requested output formats
                 logger.info(f"Generating final outputs in formats: {output_formats}")
                 generated_files = {}
                 if 'json' in output_formats:
                     try:
                         # Call the NEW function for segment-level JSON
                         persistence.save_segment_level_json(finalized_transcript, output_path_final_json)
                         generated_files['json'] = os.path.basename(output_path_final_json)
                     except Exception as e: logger.error(f"Failed to save final segment-level JSON: {e}", exc_info=True)
                 if 'srt' in output_formats:
                     try:
                         persistence.save_transcript_srt(finalized_transcript, output_path_srt)
                         generated_files['srt'] = os.path.basename(output_path_srt)
                     except Exception as e: logger.error(f"Failed to save SRT: {e}", exc_info=True)
                 if 'txt_ts' in output_formats:
                     try:
                         persistence.save_transcript_with_timestamps(finalized_transcript, output_path_timestamps_txt)
                         generated_files['txt_ts'] = os.path.basename(output_path_timestamps_txt)
                     except Exception as e: logger.error(f"Failed to save timestamped TXT: {e}", exc_info=True)

                 # Generate standard text format (requires post-processing)
                 if 'txt' in output_formats:
                     try:
                         punctuation_pipeline = models.load_punctuation_model(punctuation_model_name, processing_device)
                         final_turns = run_postprocessing(
                             identified_transcript=finalized_transcript, # Pass finalized transcript
                             punctuation_pipeline=punctuation_pipeline,
                             punctuation_chunk_size=punctuation_chunk_size,
                             cucco_instance=cucco_instance,
                             inflect_engine=inflect_engine,
                             normalize_numbers=normalize_numbers,
                             remove_fillers=remove_fillers,
                             filler_words=filler_words
                         )
                         persistence.save_final_transcript(final_turns, output_path_standard_txt)
                         generated_files['txt'] = os.path.basename(output_path_standard_txt)
                     except Exception as e: logger.error(f"Failed to save standard TXT: {e}", exc_info=True)

                 result_payload["output_files"] = generated_files # Store generated filenames
                 # END <<< MODIFICATION >>>

                 result_payload.update({ "status": "success", "message": "Full pipeline finished successfully." })
                 # Full stage complete

        # ==================================
        # === Stage: finalize_enrollment ===
        # ==================================
        elif enrollment_stage == "finalize_enrollment":
            logger.info("--- Running Finalize Enrollment Stage ---")
            try:
                # Load the potentially merged transcript from stage 1
                identified_transcript = persistence.load_word_level_transcript(temp_transcript_path) # Use renamed loader
                new_speaker_info_with_embeddings = load_enrollment_info(temp_enroll_info_path)
            except FileNotFoundError as fnf_err:
                raise RuntimeError(f"Missing intermediate data file for finalization: {fnf_err}") from fnf_err
            except Exception as load_err:
                raise RuntimeError(f"Failed to load intermediate data for finalization: {load_err}") from load_err

            if not identified_transcript:
                raise RuntimeError(f"Loaded intermediate transcript from {temp_transcript_path} is empty or invalid.")

            # ... (Determine embedding dim, enroll speakers programmatically remain the same) ...
            temp_processing_device = utils.check_device(processing_device_requested)
            orig_embedding_model_name = original_kwargs_for_result.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL)
            _, temp_emb_dim = models.load_embedding_model(orig_embedding_model_name, temp_processing_device, embedding_cache_dir)
            embedding_dim = temp_emb_dim
            logger.info(f"Determined embedding dimension for DB load: {embedding_dim}")

            logger.info(f"Loading CENTRAL speaker DB for programmatic enrollment: {central_faiss_path}, {central_map_path}")
            faiss_index = persistence.load_or_create_faiss_index(central_faiss_path, embedding_dim)
            speaker_map = persistence.load_or_create_speaker_map(central_map_path)
            db_updated_prog = False
            if new_speaker_info_with_embeddings and enrollment_map:
                if faiss_index is None or speaker_map is None:
                    logger.error("Failed to load CENTRAL FAISS index/map for programmatic enrollment. Skipping.")
                else:
                    db_updated_prog = speaker_id.enroll_speakers_programmatic(
                        enrollment_map=enrollment_map,
                        new_speaker_info=new_speaker_info_with_embeddings,
                        faiss_index=faiss_index,
                        speaker_map=speaker_map,
                        faiss_index_path=central_faiss_path,
                        speaker_map_path=central_map_path
                    )
                    result_payload["db_updated"] = db_updated_prog
            else:
                logger.info("No enrollment map provided or no new speakers identified previously. Skipping programmatic enrollment.")


            # Update Transcript with Newly Enrolled Names
            final_speaker_assignments = {}
            all_speakers_in_transcript = set()
            # Use the transcript loaded from the temp file (which might have been merged)
            for segment in identified_transcript:
                if 'words' in segment:
                    for word in segment['words']:
                        if 'speaker' in word: all_speakers_in_transcript.add(word['speaker'])

            for speaker_label in all_speakers_in_transcript:
                final_speaker_assignments[speaker_label] = enrollment_map.get(speaker_label, speaker_label)

            finalized_transcript = speaker_id.update_transcript_speakers(
                identified_transcript, final_speaker_assignments
            )

            # START <<< MODIFICATION >>> Generate requested output formats using finalized_transcript
            logger.info(f"Generating final outputs in formats: {output_formats}")
            generated_files = {}
            if 'json' in output_formats:
                try:
                    # Call the NEW function for segment-level JSON
                    persistence.save_segment_level_json(finalized_transcript, output_path_final_json)
                    generated_files['json'] = os.path.basename(output_path_final_json)
                except Exception as e: logger.error(f"Failed to save final segment-level JSON: {e}", exc_info=True)
            if 'srt' in output_formats:
                try:
                    persistence.save_transcript_srt(finalized_transcript, output_path_srt)
                    generated_files['srt'] = os.path.basename(output_path_srt)
                except Exception as e: logger.error(f"Failed to save SRT: {e}", exc_info=True)
            if 'txt_ts' in output_formats:
                try:
                    persistence.save_transcript_with_timestamps(finalized_transcript, output_path_timestamps_txt)
                    generated_files['txt_ts'] = os.path.basename(output_path_timestamps_txt)
                except Exception as e: logger.error(f"Failed to save timestamped TXT: {e}", exc_info=True)

            # Generate standard text format (requires post-processing)
            if 'txt' in output_formats:
                try:
                    # Retrieve post-processing args from original_kwargs
                    orig_punctuation_model_name = original_kwargs_for_result.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL)
                    orig_punctuation_chunk_size = original_kwargs_for_result.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE)
                    orig_normalize_numbers = original_kwargs_for_result.get('normalize_numbers', config.DEFAULT_NORMALIZE_NUMBERS)
                    orig_remove_fillers = original_kwargs_for_result.get('remove_fillers', config.DEFAULT_REMOVE_FILLERS)
                    orig_filler_words = original_kwargs_for_result.get('filler_words', config.DEFAULT_FILLER_WORDS)
                    processing_device = utils.check_device(processing_device_requested)

                    punctuation_pipeline = models.load_punctuation_model(orig_punctuation_model_name, processing_device)
                    final_turns = run_postprocessing(
                        identified_transcript=finalized_transcript, # Pass finalized transcript
                        punctuation_pipeline=punctuation_pipeline,
                        punctuation_chunk_size=orig_punctuation_chunk_size,
                        cucco_instance=cucco_instance,
                        inflect_engine=inflect_engine,
                        normalize_numbers=orig_normalize_numbers,
                        remove_fillers=orig_remove_fillers,
                        filler_words=orig_filler_words
                    )
                    persistence.save_final_transcript(final_turns, output_path_standard_txt)
                    generated_files['txt'] = os.path.basename(output_path_standard_txt)
                except Exception as e: logger.error(f"Failed to save standard TXT: {e}", exc_info=True)

            result_payload["output_files"] = generated_files # Store generated filenames
            # END <<< MODIFICATION >>>

            # Cleanup Temporary Job Files
            try:
                if os.path.exists(temp_transcript_path): os.remove(temp_transcript_path)
                if os.path.exists(temp_enroll_info_path): os.remove(temp_enroll_info_path)
                snippet_dir = os.path.join(output_dir, SNIPPETS_SUBDIR)
                if os.path.isdir(snippet_dir):
                    import shutil
                    shutil.rmtree(snippet_dir)
                    logger.info(f"Cleaned up snippets directory: {snippet_dir}")
                logger.info("Cleaned up temporary files.")
            except Exception as clean_err:
                logger.warning(f"Failed to clean up temporary files/snippets: {clean_err}")

            result_payload.update({
                "status": "success",
                "message": "Finalization and enrollment complete." if db_updated_prog else "Finalization complete (no new speakers enrolled).",
            })


    except Exception as e:
        error_message = f"Pipeline failed during stage '{enrollment_stage}': {e}"
        logger.critical(error_message, exc_info=True)
        result_payload["status"] = "error"
        result_payload["message"] = error_message
        utils.cleanup_resources(whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline, use_cuda=uses_cuda)
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
    # Input/Output
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output-dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                        help="Directory to save job-specific output files.")
    parser.add_argument("--log-file", type=str, default=config.DEFAULT_LOG_FILENAME)
    # START <<< MODIFICATION >>> Add output format selection for CLI
    parser.add_argument('--output-formats', nargs='+', default=['txt'],
                        choices=['txt', 'txt_ts', 'srt', 'json'],
                        help="Specify desired output transcript formats.")
    # END <<< MODIFICATION >>>

    # Models & Resources
    parser.add_argument("--hf-token", type=str, default=config.DEFAULT_HF_TOKEN_FILE, help="Path to Hugging Face token file.")
    parser.add_argument("--whisper-model", type=str, default=config.DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-device", type=str, default=config.DEFAULT_WHISPER_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--whisper-cache-path", type=str, default=config.WHISPER_CACHE_DIR, help="Directory to cache Whisper models.")
    parser.add_argument("--processing-device", type=str, default=config.DEFAULT_PROCESSING_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--embedding-model", type=str, default=config.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--punctuation-model", type=str, default=config.DEFAULT_PUNCTUATION_MODEL)

    # Processing Parameters
    parser.add_argument("--similarity-threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--min-segment-duration", type=float, default=config.DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--punctuation-chunk-size", type=int, default=config.DEFAULT_PUNCTUATION_CHUNK_SIZE)

    # Diarization Tuning & Merge Heuristic
    parser.add_argument('--pyannote-kwarg', action='append', nargs=2, metavar=('KEY', 'VALUE'),
                        help="Pass keyword arguments to Pyannote pipeline (e.g., clustering.threshold 0.8). "
                             "Consult Pyannote docs for valid keys. Values are parsed as float if possible, else string.")
    parser.add_argument("--enable-merge-heuristic", action='store_true', default=False,
                        help="Enable post-identification merging of short speaker segments.")
    parser.add_argument("--merge-gap-threshold-ms", type=int, default=config.DEFAULT_MERGE_GAP_THRESHOLD_MS,
                        help="Time gap threshold in milliseconds for merging speaker segments.")

    # Normalization
    parser.add_argument("--normalize-numbers", action='store_true', default=config.DEFAULT_NORMALIZE_NUMBERS)
    parser.add_argument("--remove-fillers", action='store_true', default=config.DEFAULT_REMOVE_FILLERS)

    # Logging
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose console logging (DEBUG level). Default is INFO.")

    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_arguments()
    console_log_level_cli = config.VERBOSE_CONSOLE_LOG_LEVEL if cli_args.verbose else config.DEFAULT_CONSOLE_LOG_LEVEL

    # Ensure directories exist
    try:
        os.makedirs(config.CENTRAL_DB_DIR, exist_ok=True)
        print(f"CLI: Ensured central speaker database directory exists: {config.CENTRAL_DB_DIR}")
        os.makedirs(cli_args.whisper_cache_path, exist_ok=True)
        print(f"CLI: Ensured Whisper cache directory exists: {cli_args.whisper_cache_path}")
    except OSError as e:
        print(f"CLI: WARNING - Could not create required directories: {e}")

    # Setup logging for CLI run
    cli_log_path = os.path.join(cli_args.output_dir, cli_args.log_file)
    os.makedirs(os.path.dirname(cli_log_path), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=cli_log_path,
                        filemode='w',
                        force=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level_cli)
    console_formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger = logging.getLogger()
    has_console_handler = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not has_console_handler:
        root_logger.addHandler(console_handler)

    logger.info("Starting pipeline execution from CLI...")

    # Process Pyannote kwargs from CLI
    cli_diarization_kwargs = {}
    if cli_args.pyannote_kwarg:
        for key, value_str in cli_args.pyannote_kwarg:
            try: value = float(value_str)
            except ValueError: value = value_str
            cli_diarization_kwargs[key] = value
        logger.info(f"CLI: Parsed Pyannote kwargs: {cli_diarization_kwargs}")


    result = run_full_pipeline(
        output_dir=cli_args.output_dir,
        input_audio=cli_args.input_audio,
        enrollment_stage="full",
        log_file=cli_args.log_file,
        # START <<< MODIFICATION >>> Pass output_formats from CLI
        output_formats=cli_args.output_formats,
        # END <<< MODIFICATION >>>
        hf_token_path=cli_args.hf_token,
        whisper_model_name=cli_args.whisper_model,
        whisper_device_requested=cli_args.whisper_device,
        whisper_cache_path=cli_args.whisper_cache_path,
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
        whisper_prompt="",
        whisper_language=None,
        diarization_kwargs=cli_diarization_kwargs,
        enable_merge_heuristic=cli_args.enable_merge_heuristic,
        merge_gap_threshold_ms=cli_args.merge_gap_threshold_ms
    )

    print(f"\nPipeline finished with status: {result['status']}")
    if result['status'] == 'error':
        print(f"Error message: {result['message']}")
        exit(1)
    else:
        # START <<< MODIFICATION >>> Print paths for all generated files
        print("Generated output files:")
        for format_key, filename in result.get("output_files", {}).items():
            # Construct full path for display
            full_path = os.path.join(cli_args.output_dir, filename)
            print(f"  - {format_key.upper()}: {full_path}")
        # END <<< MODIFICATION >>>
        print(f"Speaker DB Updated: {result.get('db_updated')} (in {config.CENTRAL_DB_DIR})")
        exit(0)
