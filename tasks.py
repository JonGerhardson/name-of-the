# tasks.py
import os
import time
import logging
from celery import current_task
import json
from typing import Dict, List, Any, Optional # <-- Import Dict and other necessary types

# Import the refactored pipeline function and config
import pipeline
import config

# Import the Celery app instance
from celery_app import celery as celery_app

logger = logging.getLogger(__name__)

# --- Task for Stage 1: Identification Only ---
@celery_app.task(bind=True, name='tasks.run_identification_stage')
def run_identification_stage(self, input_audio_path: str, output_dir: str, **kwargs):
    """
    Celery task for the first stage: ASR, Diarization, Identification.
    Returns info needed for potential UI enrollment, including timestamps
    and the relative path to the original audio for timestamp-based playback.
    """
    task_id = self.request.id
    logger.info(f"Stage 1 task {task_id} received for audio: {input_audio_path}")
    logger.info(f"Output directory for task {task_id}: {output_dir}")
    logger.info(f"Received kwargs for task {task_id}: {kwargs}")

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Starting Identification Stage...', 'progress': 5})

        # Prepare args, explicitly setting stage and disabling CLI enrollment
        pipeline_args = {
            "input_audio": input_audio_path,
            "output_dir": output_dir,
            "enrollment_stage": "identify_only", # <-- Set stage
            "setup_logging_in_func": False, # Celery handles logging
            # Pass through relevant overrides from web UI
            "whisper_model_name": kwargs.get('whisper_model', config.DEFAULT_WHISPER_MODEL),
            "normalize_numbers": kwargs.get('normalize_numbers', config.DEFAULT_NORMALIZE_NUMBERS), # Needed later
            "remove_fillers": kwargs.get('remove_fillers', config.DEFAULT_REMOVE_FILLERS), # Needed later
            # Add other pipeline args using defaults or kwargs
            "log_file": config.DEFAULT_LOG_FILENAME,
            "output_json_file": config.DEFAULT_IDENTIFIED_JSON_FILENAME, # Not the final one
            "output_final_file": config.DEFAULT_FINAL_OUTPUT_FILENAME, # Not used here
            "hf_token_path": config.DEFAULT_HF_TOKEN_FILE,
            "faiss_file": config.DEFAULT_FAISS_INDEX_FILENAME,
            "map_file": config.DEFAULT_SPEAKER_MAP_FILENAME,
            "whisper_device_requested": "cpu", # Keep Whisper on CPU as per original config
            "whisper_compute": kwargs.get('whisper_compute', config.DEFAULT_WHISPER_COMPUTE),
            "processing_device_requested": "cpu", # <--- FORCE CPU FOR OTHER MODELS
            "embedding_model_name": kwargs.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL), # Corrected key
            "punctuation_model_name": kwargs.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL), # Corrected key
            "similarity_threshold": kwargs.get('similarity_threshold', config.DEFAULT_SIMILARITY_THRESHOLD),
            "min_segment_duration": kwargs.get('min_segment_duration', config.DEFAULT_MIN_SEGMENT_DURATION),
            "punctuation_chunk_size": kwargs.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE),
            "filler_words": config.DEFAULT_FILLER_WORDS,
        }

        logger.info(f"Calling pipeline.run_full_pipeline (identify_only) for task {task_id}")
        self.update_state(state='PROGRESS', meta={'status': 'Running ASR/Diarization/ID...', 'progress': 10})

        # Execute Stage 1
        result = pipeline.run_full_pipeline(**pipeline_args)

        logger.info(f"Identification stage finished for task {task_id}. Status: {result.get('status')}")

        if result.get("status") in ["success", "enrollment_required"]:
            # Get the original list which includes embeddings and potentially timestamps
            unknown_speakers_full_info = result.get("unknown_speakers", [])

            # **NEW:** Extract relative path of the original audio
            # Assumes input_audio_path is within output_dir (e.g., output_dir/upload/file.wav)
            try:
                original_audio_relative_path = os.path.relpath(input_audio_path, output_dir)
                # Basic check to ensure it looks like a relative path within the job dir
                if original_audio_relative_path.startswith(".."):
                    logger.warning(f"Calculated relative path '{original_audio_relative_path}' seems outside output_dir '{output_dir}'. Using absolute path as fallback.")
                    original_audio_relative_path = None # Fallback or handle error
                else:
                     logger.info(f"Calculated relative audio path: {original_audio_relative_path}")
            except ValueError:
                logger.warning(f"Could not determine relative path for {input_audio_path} within {output_dir}. Timestamps may not work.")
                original_audio_relative_path = None # Indicate path couldn't be determined

            # Create a new list containing only the info needed by the UI
            # **MODIFIED:** Ensure start_time and end_time are included
            unknown_speakers_for_return = []
            for speaker_info in unknown_speakers_full_info:
                # Check if required keys exist
                if 'temp_id' in speaker_info and 'start_time' in speaker_info and 'end_time' in speaker_info:
                    speaker_data = {
                        'temp_id': speaker_info['temp_id'],
                        'original_label': speaker_info.get('original_label', 'N/A'), # Use .get for safety
                        'start_time': speaker_info['start_time'],
                        'end_time': speaker_info['end_time']
                    }
                    unknown_speakers_for_return.append(speaker_data)
                else:
                    logger.warning(f"Skipping speaker in return data due to missing keys (temp_id, start_time, end_time): {speaker_info}")


            # Prepare the data payload needed for both return and meta
            data_payload = {
                "status": result.get("status"), # Keep track of the pipeline function's status report
                "message": result.get("message"),
                "unknown_speakers": unknown_speakers_for_return, # Use the list with timestamps
                "temp_transcript_path": result.get("temp_transcript_path"),
                "temp_enroll_info_path": result.get("temp_enroll_info_path"),
                "output_dir": output_dir,
                "original_kwargs": result.get("original_kwargs", kwargs), # Get updated kwargs from pipeline result if available
                "original_audio_path": original_audio_relative_path # **NEW:** Add relative path
            }

            # Validate that we have the necessary path if enrollment is required
            if data_payload['status'] == 'enrollment_required' and not original_audio_relative_path:
                 logger.error(f"Enrollment required for task {task_id}, but could not determine original_audio_path. Aborting.")
                 # Update state to failure as frontend cannot proceed
                 self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': 'Could not determine path for original audio required for enrollment.'})
                 raise Exception("Could not determine path for original audio required for enrollment.")

            # Update final status for this stage
            if data_payload['status'] == 'enrollment_required':
                # Store the needed data payload in the 'meta' field for this state
                self.update_state(state='AWAITING_ENROLLMENT', meta=data_payload)
            else: # Status == 'success' (No enrollment needed)
                success_meta = {'status': data_payload.get('message','Success'), 'progress': 100}
                self.update_state(state='SUCCESS', meta=success_meta)

            # Return the payload (Celery stores this as task.result upon SUCCESS/FAILURE)
            return data_payload
        else:
            # Handle errors from the pipeline function
            error_message = result.get('message', 'Identification stage failed.')
            logger.error(f"Identification stage failed for task {task_id}: {error_message}")
            self.update_state(state='FAILURE', meta={'status': 'Identification Failed', 'error': error_message})
            raise Exception(error_message)

    except Exception as e:
        logger.error(f"Unexpected error in identification task {task_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': str(e)})
        raise e # Re-raise exception for Celery


# --- Task for Stage 2: Finalize Enrollment & Post-Processing ---
@celery_app.task(bind=True, name='tasks.run_finalization_stage')
def run_finalization_stage(self,
                           output_dir: str, # Directory containing temp files
                           enrollment_map: Dict[str, str], # {temp_id: name} from UI
                           original_kwargs: dict # Original UI options
                           ):
    """
    Celery task for the second stage: Programmatic enrollment and final processing.
    """
    task_id = self.request.id
    logger.info(f"Stage 2 task {task_id} received for output dir: {output_dir}")
    logger.info(f"Enrollment map: {enrollment_map}")
    logger.info(f"Original kwargs: {original_kwargs}")

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Starting Finalization Stage...', 'progress': 5})

        # Prepare args for the finalize stage
        pipeline_args = {
            "input_audio": None, # Set to None as it's not needed directly
            "output_dir": output_dir,
            "enrollment_stage": "finalize_enrollment", # <-- Set stage
            "enrollment_map": enrollment_map, # Pass the map from UI
            "setup_logging_in_func": False,
            # Get relevant processing options from original_kwargs
            "normalize_numbers": original_kwargs.get('normalize_numbers', config.DEFAULT_NORMALIZE_NUMBERS),
            "remove_fillers": original_kwargs.get('remove_fillers', config.DEFAULT_REMOVE_FILLERS),
            # Add other necessary args, using defaults or original_kwargs
            "log_file": config.DEFAULT_LOG_FILENAME,
            "output_json_file": config.DEFAULT_IDENTIFIED_JSON_FILENAME, # Final intermediate
            "output_final_file": config.DEFAULT_FINAL_OUTPUT_FILENAME, # Final TXT
            "hf_token_path": config.DEFAULT_HF_TOKEN_FILE, # Needed for punctuation model?
            "faiss_file": config.DEFAULT_FAISS_INDEX_FILENAME,
            "map_file": config.DEFAULT_SPEAKER_MAP_FILENAME,
            # Ensure processing device for punctuation is passed correctly
            "processing_device_requested": original_kwargs.get('processing_device_requested', config.DEFAULT_PROCESSING_DEVICE),
            "punctuation_model_name": original_kwargs.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL),
            "punctuation_chunk_size": original_kwargs.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE),
            "filler_words": original_kwargs.get('filler_words', config.DEFAULT_FILLER_WORDS),
            # Arguments NOT needed directly: whisper*, similarity*, min_segment*
            # However, embedding_model_name might be needed if reloading model for emb_dim
            "embedding_model_name": original_kwargs.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL),
        }

        logger.info(f"Calling pipeline.run_full_pipeline (finalize_enrollment) for task {task_id} with args: {pipeline_args}")
        self.update_state(state='PROGRESS', meta={'status': 'Enrolling speakers & finalizing...', 'progress': 10})

        # Execute Stage 2
        result = pipeline.run_full_pipeline(**pipeline_args)

        logger.info(f"Finalization stage finished for task {task_id}. Status: {result.get('status')}")

        if result.get("status") == "success":
            logger.info(f"Celery task {task_id} (finalization) completed successfully.")
            # For SUCCESS, meta can be simple, result holds the payload
            final_success_meta = {'status': 'Complete', 'progress': 100}
            self.update_state(state='SUCCESS', meta=final_success_meta)
            # Return final confirmation and paths
            # Celery stores this return value as task.result
            return {
                "status": "success",
                "message": result.get("message", "Processing complete!"),
                "final_transcript_filename": os.path.basename(result.get("final_transcript_path", "")),
                "intermediate_json_filename": os.path.basename(result.get("intermediate_json_path", "")),
                "log_filename": os.path.basename(result.get("log_file_path", "")),
                "db_updated": result.get('db_updated', False)
            }
        else:
            error_message = result.get('message', 'Finalization stage failed.')
            logger.error(f"Finalization stage failed for task {task_id}: {error_message}")
            self.update_state(state='FAILURE', meta={'status': 'Finalization Failed', 'error': error_message})
            # Raise exception for Celery error handling
            raise Exception(error_message)

    except Exception as e:
        logger.error(f"Unexpected error in finalization task {task_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': str(e)})
        raise e

