# tasks.py
# Updated to remove whisper_compute argument for openai-whisper compatibility
# Removed faiss_file and map_file from pipeline_args for central DB
# Added job-specific file logging setup/teardown
# Corrected Whisper cache path variable name
# Added passing of diarization_kwargs and merge heuristic controls
# Fixed merge threshold default lookup
# *** No major changes needed here as kwargs already handle passing diarization_kwargs ***
import os
import time
import logging
from celery import current_task
import json
from typing import Dict, List, Any, Optional

# Import the refactored pipeline function and config
import pipeline # Import pipeline to access its constants
import config
import log_setup # Or import specific functions

# Import the Celery app instance
from celery_app import celery as celery_app

logger = logging.getLogger(__name__)

# --- Task for Stage 1: Identification Only ---
@celery_app.task(bind=True, name='tasks.run_identification_stage')
def run_identification_stage(self, input_audio_path: str, output_dir: str, **kwargs):
    """
    Celery task for the first stage: ASR, Diarization, Identification.
    Sets up job-specific file logging.
    Accepts diarization tuning kwargs and merge heuristic controls via **kwargs.
    Accepts output_formats list via **kwargs.
    """
    task_id = self.request.id
    job_log_file_path = os.path.join(output_dir, config.DEFAULT_LOG_FILENAME)
    job_log_handler = None

    # --- Setup Job-Specific Logging ---
    try:
        root_logger = logging.getLogger()
        job_log_handler = log_setup.add_job_log_handler(
            logger_instance=root_logger,
            log_file_path=job_log_file_path,
            level=logging.DEBUG
        )
        logger.info(f"+++ Task {task_id} started: Logging to {job_log_file_path} +++")
        logger.info(f"Stage 1 task {task_id} received for audio: {input_audio_path}")
        logger.info(f"Output directory for task {task_id}: {output_dir}")
        # Log received kwargs, be mindful if they contain sensitive info
        # logger.info(f"Received kwargs for task {task_id}: {kwargs}")
        logger.debug(f"Received kwargs keys for task {task_id}: {list(kwargs.keys())}")

    except Exception as log_e:
        logger.error(f"Failed to setup job-specific logging for task {task_id} to {job_log_file_path}: {log_e}", exc_info=True)

    try: # Main task logic
        self.update_state(state='PROGRESS', meta={'status': 'Starting Identification Stage...', 'progress': 5})

        # Prepare args for run_full_pipeline using kwargs and defaults
        pipeline_args = {
            "input_audio": input_audio_path,
            "output_dir": output_dir,
            "enrollment_stage": "identify_only",
            "log_file": config.DEFAULT_LOG_FILENAME,
            # Get values from kwargs, falling back to config defaults if not present
            "output_formats": kwargs.get('output_formats', ['txt']), # Default to txt if not provided
            "hf_token_path": config.DEFAULT_HF_TOKEN_FILE, # Usually not passed via UI
            "whisper_model_name": kwargs.get('whisper_model', config.DEFAULT_WHISPER_MODEL), # 'whisper_model' is the key from web_ui.py
            "whisper_device_requested": kwargs.get('whisper_device_requested', config.DEFAULT_WHISPER_DEVICE),
            "whisper_cache_path": kwargs.get('whisper_cache_path', config.WHISPER_CACHE_DIR),
            "processing_device_requested": kwargs.get('processing_device_requested', config.DEFAULT_PROCESSING_DEVICE),
            "embedding_model_name": kwargs.get('embedding_model_name', config.DEFAULT_EMBEDDING_MODEL),
            "punctuation_model_name": kwargs.get('punctuation_model_name', config.DEFAULT_PUNCTUATION_MODEL),
            "similarity_threshold": kwargs.get('similarity_threshold', config.DEFAULT_SIMILARITY_THRESHOLD),
            "min_segment_duration": kwargs.get('min_segment_duration', config.DEFAULT_MIN_SEGMENT_DURATION),
            "punctuation_chunk_size": kwargs.get('punctuation_chunk_size', config.DEFAULT_PUNCTUATION_CHUNK_SIZE),
            "normalize_numbers": kwargs.get('normalize_numbers', config.DEFAULT_NORMALIZE_NUMBERS),
            "remove_fillers": kwargs.get('remove_fillers', config.DEFAULT_REMOVE_FILLERS),
            "filler_words": kwargs.get('filler_words', config.DEFAULT_FILLER_WORDS),
            "whisper_prompt": kwargs.get('whisper_prompt', ''),
            "whisper_language": kwargs.get('whisper_language', None),
            # --- This is where the dict created in web_ui.py is passed ---
            "diarization_kwargs": kwargs.get('diarization_kwargs', None), # Pass the dict or None
            "enable_merge_heuristic": kwargs.get('enable_merge_heuristic', False),
            "merge_gap_threshold_ms": kwargs.get('merge_gap_threshold_ms', config.DEFAULT_MERGE_GAP_THRESHOLD_MS)
        }

        logger.info(f"Calling pipeline.run_full_pipeline (identify_only) for task {task_id}")
        logger.debug(f"Pipeline args for task {task_id}: {pipeline_args}") # Log the final args being passed
        self.update_state(state='PROGRESS', meta={'status': 'Running ASR/Diarization/ID...', 'progress': 10})

        # Execute Stage 1
        result = pipeline.run_full_pipeline(**pipeline_args)

        logger.info(f"Identification stage finished for task {task_id}. Status: {result.get('status')}")

        # Process result
        if result.get("status") in ["success", "enrollment_required"]:
            # Calculate relative path for audio playback in enrollment
            original_audio_relative_path = None
            try:
                # Ensure both paths are absolute for reliable relpath calculation
                abs_output_dir = os.path.abspath(output_dir)
                abs_input_audio_path = os.path.abspath(input_audio_path)

                # Check if input is already within output (e.g., in upload subdir)
                if abs_input_audio_path.startswith(abs_output_dir):
                     original_audio_relative_path = os.path.relpath(abs_input_audio_path, abs_output_dir)
                     logger.info(f"Calculated relative audio path (within output): {original_audio_relative_path}")
                else:
                     # This case might indicate an issue if enrollment playback relies on relative paths
                     logger.warning(f"Input audio path '{abs_input_audio_path}' is outside the job output directory '{abs_output_dir}'. Relative path calculation might be problematic for UI playback.")
                     # Fallback or specific handling might be needed depending on deployment
                     original_audio_relative_path = None # Or keep absolute path if UI can handle it

            except ValueError as e:
                logger.warning(f"Could not determine relative path for {input_audio_path} within {output_dir}: {e}. Timestamps/Playback might not work.")
                original_audio_relative_path = None

            # Prepare payload for the frontend status check
            data_payload = {
                "status": result.get("status"),
                "message": result.get("message"),
                "unknown_speakers": result.get("unknown_speakers", []),
                "temp_transcript_path": result.get("temp_transcript_path"),
                "temp_enroll_info_path": result.get("temp_enroll_info_path"),
                "output_dir": output_dir, # Needed for stage 2
                # Pass back the original kwargs used, including the selected output formats and tuning params
                "original_kwargs": pipeline_args, # Pass back the actual args used
                "original_audio_path": original_audio_relative_path, # Relative path for UI playback
                "log_filename": os.path.basename(job_log_file_path) if job_log_handler else None
                # NOTE: output_files dict is only populated in stage 2 / full run
            }

            # Specific check for enrollment requirement and audio path
            if data_payload['status'] == 'enrollment_required' and data_payload['original_audio_path'] is None:
                 logger.error(f"Enrollment required for task {task_id}, but could not determine original_audio_path relative to output dir. Aborting.")
                 error_msg = 'Could not determine path for original audio required for enrollment playback.'
                 self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': error_msg})
                 raise ValueError(error_msg)

            # Update task state based on pipeline outcome
            if data_payload['status'] == 'enrollment_required':
                self.update_state(state='AWAITING_ENROLLMENT', meta=data_payload)
            else: # Stage 1 success without enrollment
                success_meta = {'status': data_payload.get('message','Success - No enrollment needed'), 'progress': 100}
                # Although unlikely for identify_only, include output files if present
                if result.get("output_files"):
                    success_meta["output_files"] = result["output_files"]
                self.update_state(state='SUCCESS', meta=success_meta)

            # Return the payload which becomes task.result
            return data_payload
        else:
            # Pipeline returned an error status
            error_message = result.get('message', 'Identification stage failed.')
            logger.error(f"Identification stage failed for task {task_id}: {error_message}")
            self.update_state(state='FAILURE', meta={'status': 'Identification Failed', 'error': error_message})
            raise RuntimeError(f"Identification stage failed: {error_message}")

    except Exception as e:
        error_str = str(e)
        logger.error(f"Unexpected error in identification task {task_id}: {error_str}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': error_str})
        raise e # Reraise exception for Celery
    finally:
        # --- Cleanup Job-Specific Logging ---
        if job_log_handler:
            logger.info(f"--- Task {task_id} finished: Removing job log handler ---")
            log_setup.remove_job_log_handler(logging.getLogger(), job_log_handler)
        else:
            logger.info(f"--- Task {task_id} finished (no job log handler to remove) ---")


# --- Task for Stage 2: Finalize Enrollment & Post-Processing ---
@celery_app.task(bind=True, name='tasks.run_finalization_stage')
def run_finalization_stage(self,
                           output_dir: str,
                           enrollment_map: Dict[str, str],
                           original_kwargs: dict # Contains all args from stage 1, including tuning params & output_formats
                           ):
    """
    Celery task for the second stage: Programmatic enrollment and final processing.
    Sets up job-specific file logging. Generates final output formats based on original_kwargs.
    """
    task_id = self.request.id
    job_log_file_path = os.path.join(output_dir, config.DEFAULT_LOG_FILENAME)
    job_log_handler = None

    # --- Setup Job-Specific Logging ---
    try:
        root_logger = logging.getLogger()
        job_log_handler = log_setup.add_job_log_handler(
            logger_instance=root_logger,
            log_file_path=job_log_file_path,
            level=logging.DEBUG
        )
        logger.info(f"+++ Task {task_id} (Stage 2) started: Logging to {job_log_file_path} +++")
        logger.info(f"Stage 2 task {task_id} received for output dir: {output_dir}")
        logger.info(f"Enrollment map: {enrollment_map}")
        logger.debug(f"Original kwargs for task {task_id}: {original_kwargs}")

    except Exception as log_e:
        logger.error(f"Failed to setup job-specific logging for task {task_id} (Stage 2) to {job_log_file_path}: {log_e}", exc_info=True)

    try: # Main task logic
        self.update_state(state='PROGRESS', meta={'status': 'Starting Finalization Stage...', 'progress': 5})

        # Prepare args for run_full_pipeline using original_kwargs passed from stage 1
        # The pipeline function knows how to handle 'finalize_enrollment' stage
        # We just need to ensure all necessary keys are present in original_kwargs
        pipeline_args = {
            **original_kwargs, # Start with all original arguments
            "input_audio": None, # Not needed for finalization
            "output_dir": output_dir,
            "enrollment_stage": "finalize_enrollment",
            "enrollment_map": enrollment_map,
            "log_file": config.DEFAULT_LOG_FILENAME,
            # output_formats should already be in original_kwargs
        }

        # Remove keys that might confuse the finalize stage if they exist from stage 1 kwargs
        pipeline_args.pop('input_audio_path', None) # Use input_audio=None instead

        logger.info(f"Calling pipeline.run_full_pipeline (finalize_enrollment) for task {task_id}")
        logger.debug(f"Pipeline args for task {task_id} (Stage 2): {pipeline_args}")
        self.update_state(state='PROGRESS', meta={'status': 'Enrolling speakers & finalizing...', 'progress': 10})

        # Execute Stage 2
        result = pipeline.run_full_pipeline(**pipeline_args)

        logger.info(f"Finalization stage finished for task {task_id}. Status: {result.get('status')}")

        # Process result
        if result.get("status") == "success":
            logger.info(f"Celery task {task_id} (finalization) completed successfully.")
            final_success_meta = {
                'status': result.get("message", "Processing complete!"),
                'progress': 100,
                # Include the list of generated files for the UI
                "output_files": result.get("output_files", {}), # Dict of format:filename
                "log_filename": os.path.basename(job_log_file_path) if job_log_handler else None,
                "db_updated": result.get('db_updated', False)
            }
            self.update_state(state='SUCCESS', meta=final_success_meta)
            # Return the payload which becomes task.result
            return final_success_meta
        else:
            # Pipeline returned an error status
            error_message = result.get('message', 'Finalization stage failed.')
            logger.error(f"Finalization stage failed for task {task_id}: {error_message}")
            self.update_state(state='FAILURE', meta={'status': 'Finalization Failed', 'error': error_message})
            raise RuntimeError(f"Finalization stage failed: {error_message}")

    except Exception as e:
        error_str = str(e)
        logger.error(f"Unexpected error in finalization task {task_id}: {error_str}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'Task Error', 'error': error_str})
        raise e # Reraise exception for Celery
    finally:
        # --- Cleanup Job-Specific Logging ---
        if job_log_handler:
            logger.info(f"--- Task {task_id} (Stage 2) finished: Removing job log handler ---")
            log_setup.remove_job_log_handler(logging.getLogger(), job_log_handler)
        else:
            logger.info(f"--- Task {task_id} (Stage 2) finished (no job log handler to remove) ---")


