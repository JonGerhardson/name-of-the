# tasks.py
# Updated to remove whisper_compute argument for openai-whisper compatibility
# Removed faiss_file and map_file from pipeline_args for central DB
# Integrated whisper_prompt functionality
# Integrated whisper_language functionality
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
    Uses the central speaker database defined in config.py.
    Accepts optional 'whisper_prompt' and 'whisper_language' kwargs. # <-- Updated docstring
    """
    task_id = self.request.id
    logger.info(f"Stage 1 task {task_id} received for audio: {input_audio_path}")
    logger.info(f"Output directory for task {task_id}: {output_dir}")
    logger.info(f"Received kwargs for task {task_id}: {kwargs}") # Prompt & lang will be visible here

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Starting Identification Stage...', 'progress': 5})

        # Prepare args, explicitly setting stage and disabling CLI enrollment
        pipeline_args = {
            "input_audio": input_audio_path,
            "output_dir": output_dir,
            "enrollment_stage": "identify_only", # <-- Set stage
            "setup_logging_in_func": False, # Celery handles logging
            # Pass through relevant overrides from web UI / kwargs
            "whisper_model_name": kwargs.get('whisper_model', config.DEFAULT_WHISPER_MODEL),
            "normalize_numbers": kwargs.get('normalize_numbers', config.DEFAULT_NORMALIZE_NUMBERS),
            "remove_fillers": kwargs.get('remove_fillers', config.DEFAULT_REMOVE_FILLERS),
            "log_file": config.DEFAULT_LOG_FILENAME, # Job specific log
            "output_json_file": config.DEFAULT_IDENTIFIED_JSON_FILENAME, # Job specific intermediate output
            "output_final_file": config.DEFAULT_FINAL_OUTPUT_FILENAME, # Job specific final output
            "hf_token_path": config.DEFAULT_HF_TOKEN_FILE,
            "whisper_device_requested": kwargs.get('whisper_device_requested', config.DEFAULT_WHISPER_DEVICE), # Pass from kwargs or use config default
            "processing_device_requested": kwargs.get('processing_device_requested', config.DEFAULT_PROCESSING_DEVICE), # Pass from kwargs or use config default
            "embedding_model_name": kwargs.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL),
            "punctuation_model_name": kwargs.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL),
            "similarity_threshold": kwargs.get('similarity_threshold', config.DEFAULT_SIMILARITY_THRESHOLD),
            "min_segment_duration": kwargs.get('min_segment_duration', config.DEFAULT_MIN_SEGMENT_DURATION),
            "punctuation_chunk_size": kwargs.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE),
            "filler_words": kwargs.get('filler_words', config.DEFAULT_FILLER_WORDS), # Allow override from kwargs if needed
            "whisper_prompt": kwargs.get('whisper_prompt', ''), # Get prompt from kwargs, default empty
            # START <<< INTEGRATED UPDATE >>> Pass language to pipeline function
            "whisper_language": kwargs.get('whisper_language', None) # Pass language (None for auto-detect in pipeline)
            # END <<< INTEGRATED UPDATE >>>
        }

        logger.info(f"Calling pipeline.run_full_pipeline (identify_only) for task {task_id}")
        self.update_state(state='PROGRESS', meta={'status': 'Running ASR/Diarization/ID...', 'progress': 10})

        # Execute Stage 1
        result = pipeline.run_full_pipeline(**pipeline_args)

        logger.info(f"Identification stage finished for task {task_id}. Status: {result.get('status')}")

        if result.get("status") in ["success", "enrollment_required"]:
            # Get the original list which includes embeddings and potentially timestamps
            # Note: This info comes from the pipeline result, which processed it correctly
            unknown_speakers_full_info = result.get("unknown_speakers", [])

            # Extract relative path of the original audio (logic remains the same)
            original_audio_relative_path = None
            try:
                original_audio_relative_path = os.path.relpath(input_audio_path, output_dir)
                if original_audio_relative_path.startswith(".."):
                    logger.warning(f"Calculated relative path '{original_audio_relative_path}' seems outside output_dir '{output_dir}'. Using absolute path as fallback.")
                    original_audio_relative_path = None # Fallback to None which will be handled later
                else:
                    logger.info(f"Calculated relative audio path: {original_audio_relative_path}")
            except ValueError:
                logger.warning(f"Could not determine relative path for {input_audio_path} within {output_dir}. Timestamps may not work.")
                original_audio_relative_path = None # Ensure it's None if error

            # Prepare the data payload needed for both return and meta (logic remains the same)
            data_payload = {
                "status": result.get("status"),
                "message": result.get("message"),
                "unknown_speakers": result.get("unknown_speakers", []), # Pass the already processed list for UI
                "temp_transcript_path": result.get("temp_transcript_path"),
                "temp_enroll_info_path": result.get("temp_enroll_info_path"),
                "output_dir": output_dir,
                # Store original kwargs, INCLUDING the prompt and language if they were sent
                "original_kwargs": result.get("original_kwargs", kwargs),
                "original_audio_path": original_audio_relative_path
            }

            # Validate path if enrollment required (logic remains the same)
            if data_payload['status'] == 'enrollment_required' and not original_audio_relative_path:
                logger.error(f"Enrollment required for task {task_id}, but could not determine original_audio_path relative to output dir. Aborting.")
                self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': 'Could not determine path for original audio required for enrollment.'})
                # Raising an exception here is better for Celery to mark failure clearly
                raise ValueError("Could not determine path for original audio required for enrollment.")

            # Update final status for this stage (logic remains the same)
            if data_payload['status'] == 'enrollment_required':
                self.update_state(state='AWAITING_ENROLLMENT', meta=data_payload)
            else: # Status == 'success'
                success_meta = {'status': data_payload.get('message','Success'), 'progress': 100}
                self.update_state(state='SUCCESS', meta=success_meta)

            # Return the payload
            return data_payload
        else:
            # Handle errors from the pipeline function
            error_message = result.get('message', 'Identification stage failed.')
            logger.error(f"Identification stage failed for task {task_id}: {error_message}")
            self.update_state(state='FAILURE', meta={'status': 'Identification Failed', 'error': error_message})
            # Raising an exception here is better for Celery to mark failure clearly
            raise RuntimeError(f"Identification stage failed: {error_message}")

    except Exception as e:
        # Catch any unexpected error, including the ValueError raised above
        error_str = str(e)
        logger.error(f"Unexpected error in identification task {task_id}: {error_str}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': error_str})
        # Re-raise the original exception to ensure Celery handles it correctly
        raise e


# --- Task for Stage 2: Finalize Enrollment & Post-Processing ---
# NOTE: No changes needed here for language selection, as ASR is done in Stage 1.
# The language used is stored in original_kwargs if needed for logging/reference.
@celery_app.task(bind=True, name='tasks.run_finalization_stage')
def run_finalization_stage(self,
                             output_dir: str, # Directory containing temp files
                             enrollment_map: Dict[str, str], # {temp_id: name} from UI
                             original_kwargs: dict # Original UI options (including prompt/lang if sent)
                             ):
    """
    Celery task for the second stage: Programmatic enrollment and final processing.
    Uses the central speaker database defined in config.py.
    """
    task_id = self.request.id
    logger.info(f"Stage 2 task {task_id} received for output dir: {output_dir}")
    logger.info(f"Enrollment map: {enrollment_map}")
    logger.info(f"Original kwargs received: {original_kwargs}") # Prompt & lang might be visible here, but aren't used

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Starting Finalization Stage...', 'progress': 5})

        # Prepare args for the finalize stage
        # Note: whisper_prompt/whisper_language are in original_kwargs but not needed/passed here.
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
            "log_file": config.DEFAULT_LOG_FILENAME, # Job specific log
            "output_json_file": config.DEFAULT_IDENTIFIED_JSON_FILENAME, # Job specific final intermediate
            "output_final_file": config.DEFAULT_FINAL_OUTPUT_FILENAME, # Job specific final TXT
            "hf_token_path": config.DEFAULT_HF_TOKEN_FILE,
            "processing_device_requested": original_kwargs.get('processing_device_requested', config.DEFAULT_PROCESSING_DEVICE),
            "punctuation_model_name": original_kwargs.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL),
            "punctuation_chunk_size": original_kwargs.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE),
            "filler_words": original_kwargs.get('filler_words', config.DEFAULT_FILLER_WORDS),
            # Arguments needed indirectly (for loading models/DB): embedding_model_name
            "embedding_model_name": original_kwargs.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL),
            # Pass similarity threshold and duration in case they are needed for future logic or reloading db
            "similarity_threshold": original_kwargs.get('similarity_threshold', config.DEFAULT_SIMILARITY_THRESHOLD),
            "min_segment_duration": original_kwargs.get('min_segment_duration', config.DEFAULT_MIN_SEGMENT_DURATION),
        }

        logger.info(f"Calling pipeline.run_full_pipeline (finalize_enrollment) for task {task_id}")
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
            raise RuntimeError(f"Finalization stage failed: {error_message}")

    except Exception as e:
        error_str = str(e)
        logger.error(f"Unexpected error in finalization task {task_id}: {error_str}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': error_str})
        raise e
