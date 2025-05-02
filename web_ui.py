# web_ui.py
import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, abort, current_app
import tasks # Import your Celery tasks module
from celery_app import celery # Import the celery instance
import config # Import config for defaults
from werkzeug.utils import secure_filename

# --- Configuration ---
# These are defaults, Flask config below is the primary source within the app context
# UPLOAD_FOLDER = '/app/audio_input' # Not directly used if saving to job folders
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
# OUTPUT_FOLDER = '/app/transcripts_output' # Use config.OUTPUT_FOLDER
# SNIPPETS_SUBDIR = "snippets" # Use config.SNIPPETS_SUBDIR

# --- App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['OUTPUT_FOLDER'] = config.DEFAULT_OUTPUT_DIR # Load from config
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_CHANGE_ME')

# --- Logging Setup ---
# Configure logging level based on Flask's debug mode (set in app.run)
# Use a more detailed format for debug mode
log_format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
# Configure root logger or Flask's logger
logging.basicConfig(level=logging.DEBUG, format=log_format) # Set basicConfig level to DEBUG to capture everything initially
# Get Flask's logger
logger = logging.getLogger('flask.app') # Use Flask's standard logger name
# Adjust level based on app.debug after app context is available or before app.run
# Note: app.debug is True when running app.run(debug=True)

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    current_app.logger.info("Serving index page.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and starts the STAGE 1 processing task."""
    current_app.logger.debug("Received request to /upload")
    if 'file' not in request.files:
        current_app.logger.warning("Upload attempt with no file part.")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        current_app.logger.warning("Upload attempt with no selected file.")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())
        # --- Create the main job output directory ---
        job_output_dir = os.path.join(current_app.config['OUTPUT_FOLDER'], job_id)
        # --- Create a specific subdirectory for the original upload ---
        upload_subdir = os.path.join(job_output_dir, "upload") # Good practice

        try:
            # Create both directories
            os.makedirs(upload_subdir, exist_ok=True) # Creates parent job_output_dir too
            original_filename = secure_filename(file.filename)
            save_path = os.path.join(upload_subdir, original_filename)
            file.save(save_path)
            current_app.logger.info(f"Job {job_id}: File '{original_filename}' saved to {save_path}")
        except OSError as e:
            current_app.logger.error(f"Job {job_id}: Error creating directories or saving file: {e}", exc_info=True)
            return jsonify({'error': 'Server error creating directories or saving file'}), 500
        except Exception as e:
            current_app.logger.error(f"Job {job_id}: Error saving uploaded file: {e}", exc_info=True)
            return jsonify({'error': 'Server error saving file'}), 500

        try:
            # Get optional arguments from form data
            whisper_model = request.form.get('whisper_model', config.DEFAULT_WHISPER_MODEL)
            normalize_numbers_str = request.form.get('normalize_numbers', 'false')
            normalize_numbers = normalize_numbers_str.lower() == 'true'
            remove_fillers_str = request.form.get('remove_fillers', 'false')
            remove_fillers = remove_fillers_str.lower() == 'true'
            # processing_device = request.form.get('processing_device', config.DEFAULT_PROCESSING_DEVICE) # Example if you add device selection
            current_app.logger.debug(f"Job {job_id}: Form options - whisper_model={whisper_model}, normalize_numbers={normalize_numbers}, remove_fillers={remove_fillers}")

            # Collect all relevant pipeline kwargs from form or config
            task_kwargs = {
                'whisper_model': whisper_model,
                'whisper_compute': config.DEFAULT_WHISPER_COMPUTE,
                'embedding_model_name': config.DEFAULT_EMBEDDING_MODEL,
                'punctuation_model_name': config.DEFAULT_PUNCTUATION_MODEL,
                'similarity_threshold': config.DEFAULT_SIMILARITY_THRESHOLD,
                'min_segment_duration': config.DEFAULT_MIN_SEGMENT_DURATION,
                'punctuation_chunk_size': config.DEFAULT_PUNCTUATION_CHUNK_SIZE,
                'normalize_numbers': normalize_numbers,
                'remove_fillers': remove_fillers,
                'filler_words': config.DEFAULT_FILLER_WORDS,
                'processing_device_requested': "cpu", # Force CPU for now, or get from form
                # Note: whisper_device is handled separately in the task based on its specific setting
            }
            current_app.logger.debug(f"Job {job_id}: Prepared task_kwargs: {task_kwargs}")

            # --- Pass the *saved path* of the uploaded file ---
            task = tasks.run_identification_stage.delay(
                input_audio_path=save_path, # Use the path where the file was actually saved
                output_dir=job_output_dir,  # Pass the main job output directory
                **task_kwargs
            )
            current_app.logger.info(f"Job {job_id}: Started Stage 1 Celery task {task.id}")
            return jsonify({'task_id': task.id, 'job_id': job_id}), 202 # Accepted

        except Exception as e:
            current_app.logger.error(f"Job {job_id}: Error during task start: {e}", exc_info=True)
            # Consider cleanup: remove job_output_dir if task fails to start?
            return jsonify({'error': 'Server error during task start'}), 500
    else:
        current_app.logger.warning(f"Upload attempt with disallowed file type: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/status/<task_id>')
def task_status(task_id):
    """Checks the status of a Celery task (could be Stage 1 or Stage 2)."""
    current_app.logger.debug(f"Checking status for task: {task_id}")
    try:
        task = celery.AsyncResult(task_id)
        response = {
            'task_id': task_id,
            'state': task.state,
            'result': None, # Will be populated on success/awaiting
            'status': task.state, # Default status message
            'progress': None,
            'error': None
        }
        task_meta = task.info if isinstance(task.info, dict) else {}
        task_return_value = task.result # Actual return value from task function

        # --- State Handling ---
        if task.state == 'PENDING':
            response['status'] = 'Pending...'
        elif task.state == 'STARTED':
            response['status'] = task_meta.get('status', 'Started...')
        elif task.state == 'PROGRESS':
             response['status'] = task_meta.get('status', 'Processing...')
             response['progress'] = task_meta.get('progress', 0)
        elif task.state == 'AWAITING_ENROLLMENT': # Explicit state set by task
             response['status'] = task_meta.get('message', 'Enrollment Required')
             # IMPORTANT: Pass the *meta* dictionary as the result for this state
             # because the task function doesn't 'return' in this state, it updates state.
             response['result'] = task_meta
             if 'unknown_speakers' not in response['result']:
                 current_app.logger.warning(f"Task {task_id} AWAITING_ENROLLMENT but 'unknown_speakers' missing in meta.")
        elif task.state == 'SUCCESS':
            # --- START RE-CORRECTION ---
            # On SUCCESS, always pass the actual task return value (task.result)
            # back to the frontend in the 'result' field.
            response['result'] = task_return_value
            response['progress'] = 100 # Mark progress as complete

            # Set a default status message, can be overridden by message in result
            response['status'] = 'Complete' # Default, will be overridden below if message exists
            if isinstance(task_return_value, dict):
                # Use message from result if available, otherwise keep 'Complete'
                response['status'] = task_return_value.get('message', response['status'])
            elif task_return_value is None:
                 # Handle case where task succeeds but returns None (shouldn't happen ideally)
                 current_app.logger.warning(f"Task {task_id} succeeded but returned None result.")
            # --- END RE-CORRECTION ---

        elif task.state == 'FAILURE':
            error_info = str(task.info) if task.info else "Unknown error"
            response['status'] = task_meta.get('status', 'Failed') # Use meta status if available
            response['error'] = error_info
            response['result'] = {'error': error_info} # Store error in result for JS
            current_app.logger.error(f"Task {task_id} failed: {task.info}", exc_info=isinstance(task.info, Exception)) # Log traceback if info is exception
        else: # Handle other states like RETRY, REVOKED
            response['status'] = task.state

        current_app.logger.debug(f"Status for task {task_id}: {response}")
        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Error checking status for task {task_id}: {e}", exc_info=True)
        return jsonify({
            'task_id': task_id, 'state': 'FAILURE',
            'status': 'Error checking task status', 'error': str(e)
        }), 500

# --- Endpoint to handle enrollment submission ---
@app.route('/enroll', methods=['POST'])
def enroll_speakers():
    """Receives enrollment data and starts the Stage 2 finalization task."""
    current_app.logger.debug("Received request to /enroll")
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("Enrollment request with missing JSON data.")
            return jsonify({'error': 'Missing JSON data'}), 400

        job_id = data.get('job_id')
        output_dir = data.get('output_dir')
        enrollment_map = data.get('enrollment_map')
        original_kwargs = data.get('original_kwargs')
        current_app.logger.debug(f"Enrollment data received: job_id={job_id}, output_dir={output_dir}, map_keys={list(enrollment_map.keys()) if enrollment_map else None}")

        # --- Validate required data ---
        missing = []
        if not job_id: missing.append('job_id')
        if not output_dir: missing.append('output_dir')
        if enrollment_map is None: missing.append('enrollment_map') # Allow empty map {}
        if original_kwargs is None: missing.append('original_kwargs') # Allow empty dict {}

        if missing:
            current_app.logger.warning(f"Enrollment request missing data. Missing: {missing}")
            return jsonify({'error': f'Missing required data for enrollment: {", ".join(missing)}'}), 400

        # --- Security Check: Ensure output_dir is within the main output folder ---
        abs_output_folder = os.path.abspath(current_app.config['OUTPUT_FOLDER'])
        abs_output_dir = os.path.abspath(output_dir)

        # Check 1: Is output_dir truly inside OUTPUT_FOLDER?
        # Check 2: Does the provided job_id match the last part of the output_dir path?
        if not abs_output_dir.startswith(abs_output_folder) or os.path.basename(abs_output_dir) != job_id:
            current_app.logger.error(f"Potential path manipulation attempt or mismatch for job {job_id}, output_dir {output_dir}")
            return jsonify({'error': 'Invalid output directory or job ID mismatch'}), 400

        current_app.logger.info(f"Job {job_id}: Received enrollment request with map: {enrollment_map}")

        # --- Start Stage 2 Task ---
        finalize_task = tasks.run_finalization_stage.delay(
            output_dir=output_dir, # Use the validated output_dir
            enrollment_map=enrollment_map,
            original_kwargs=original_kwargs
        )
        current_app.logger.info(f"Job {job_id}: Started Stage 2 (Finalization) Celery task {finalize_task.id}")
        return jsonify({'finalize_task_id': finalize_task.id}), 202 # Accepted

    except Exception as e:
        current_app.logger.error(f"Error processing enrollment request: {e}", exc_info=True)
        return jsonify({'error': 'Server error processing enrollment'}), 500


@app.route('/results/<job_id>/<path:filename>') # Use <path:filename> to handle potential subdirs if needed
def get_result_file(job_id, filename):
    """Serves a specific result file associated with a completed job."""
    current_app.logger.debug(f"Result file request: job_id={job_id}, filename={filename}")
    # --- Basic Security ---
    secure_job_id = secure_filename(job_id)
    # Filename might contain valid characters like '.', '_', '-' which secure_filename allows
    # but we rely more on the path joining and send_from_directory's security
    # secure_file = secure_filename(filename) # Apply basic cleaning - not needed if using filename directly below

    # --- More Robust Path Checks ---
    if secure_job_id != job_id: # Ensure job_id wasn't weirdly modified
        current_app.logger.warning(f"Invalid characters detected in job ID {job_id}")
        abort(400, description="Invalid job ID format.")
    if '..' in filename or filename.startswith('/'):
        current_app.logger.warning(f"Path traversal attempt detected for job {job_id}, filename {filename}")
        abort(400, description="Invalid filename path.")

    try:
        # --- Construct the FULL path to the JOB directory ---
        job_directory = os.path.join(current_app.config['OUTPUT_FOLDER'], secure_job_id)

        # --- Security Check: Ensure the resolved job directory is within the main output folder ---
        abs_output_folder = os.path.abspath(current_app.config['OUTPUT_FOLDER'])
        abs_job_directory = os.path.abspath(job_directory)

        if not abs_job_directory.startswith(abs_output_folder):
            current_app.logger.error(f"Attempt to access directory outside designated output folder: {job_directory}")
            abort(404, description="Result file not found (invalid path).")

        # --- Log the exact path being checked ---
        full_file_path = os.path.join(abs_job_directory, filename) # Use original filename for path check
        current_app.logger.info(f"Attempting to serve file: '{filename}' from directory: '{abs_job_directory}'")
        current_app.logger.debug(f"Checking existence of full path: '{full_file_path}'")

        # --- Check if file actually exists before sending ---
        if not os.path.isfile(full_file_path):
            current_app.logger.warning(f"Result file does not exist at path: {full_file_path}")
            abort(404, description="Result file not found on server.")

        # --- Serve the file ---
        return send_from_directory(
            directory=abs_job_directory, # Use absolute path for safety
            path=filename, # The filename requested
            as_attachment=False # Allow viewing in browser for text/json
        )

    except FileNotFoundError: # Should be caught by os.path.isfile, but keep for safety
        # Construct path again for logging if needed - handle potential error if filename is None
        log_path = "N/A"
        if filename:
            try:
                 log_path = os.path.join(current_app.config['OUTPUT_FOLDER'], secure_job_id, filename)
            except Exception:
                pass # Keep log_path as N/A
        current_app.logger.warning(f"send_from_directory failed for: {filename} in job {job_id} (Path: {log_path})")
        abort(404, description="Result file not found.")
    except Exception as e:
        current_app.logger.error(f"Error serving result file for job {job_id}, filename {filename}: {e}", exc_info=True)
        abort(500, description="Error retrieving result file.")


# --- Route for Snippets ---
@app.route('/snippets/<job_id>/<temp_id>')
def get_speaker_snippet(job_id, temp_id):
    """Serves a specific speaker audio snippet file."""
    current_app.logger.debug(f"Snippet request received: job_id={job_id}, temp_id={temp_id}")

    # --- Basic Security ---
    secure_job_id = secure_filename(job_id)
    # Assume temp_id (e.g., 'UNKNOWN_1') is safe as generated by the backend.
    # Do basic check for path traversal in temp_id.
    if '..' in temp_id or temp_id.startswith('/'):
         current_app.logger.warning(f"Path traversal attempt detected in temp_id: {temp_id}")
         abort(400, description="Invalid speaker ID path.")

    # Check job_id validity after securing it
    if secure_job_id != job_id:
        current_app.logger.warning(f"Invalid characters detected in snippet request job_id: {job_id}")
        abort(400, description="Invalid job ID format.")

    # --- Define filename variable outside try for use in except block if needed ---
    # Check if temp_id already ends with .wav (or other expected audio ext)
    # This handles the case where the route might unexpectedly receive 'UNKNOWN_1.wav'
    if temp_id.lower().endswith(('.wav', '.mp3', '.flac')): # Add other expected snippet formats if needed
        # If it already ends with expected extension, use it directly but prepend snippet_
        # Ensure 'snippet_' prefix is correct based on how files are saved
        filename = f"snippet_{temp_id}"
        current_app.logger.debug(f"temp_id '{temp_id}' already has extension, using filename: '{filename}'")
    else:
        # Otherwise, assume it's just the ID (e.g., 'UNKNOWN_1') and append .wav
        filename = f"snippet_{temp_id}.wav"
        current_app.logger.debug(f"temp_id '{temp_id}' needs extension, using filename: '{filename}'")


    try: # <<< TRY BLOCK STARTS HERE >>>
        # --- Construct the FULL path to the SNIPPETS directory ---
        base_output_folder = current_app.config['OUTPUT_FOLDER']
        snippet_subdir_name = getattr(config, 'SNIPPETS_SUBDIR', 'snippets')
        snippet_dir = os.path.join(base_output_folder, secure_job_id, snippet_subdir_name)

        # --- Security Check: Ensure the resolved snippet directory is valid ---
        abs_output_folder = os.path.abspath(base_output_folder)
        abs_snippet_dir = os.path.abspath(snippet_dir)

        if not abs_snippet_dir.startswith(abs_output_folder):
            current_app.logger.error(f"Attempt to access snippets directory outside designated output folder: {snippet_dir}")
            abort(404, description="Snippet not found (invalid path).")

        # --- Log the exact path being checked ---
        full_file_path = os.path.join(abs_snippet_dir, filename) # Use the filename constructed above
        current_app.logger.info(f"Attempting to serve snippet: '{filename}' from directory: '{abs_snippet_dir}'")
        current_app.logger.debug(f"Checking existence of full path: '{full_file_path}'")

        # --- Check if file actually exists before sending ---
        if not os.path.isfile(full_file_path):
            current_app.logger.warning(f"Snippet file does not exist at path: {full_file_path}")
            abort(404, description="Snippet file not found on server.")
        else:
            current_app.logger.info(f"Snippet file FOUND at path: {full_file_path}. Proceeding to serve.")

        # --- Serve the file ---
        # Determine mimetype based on filename extension
        mimetype = 'application/octet-stream' # Default
        if filename.lower().endswith('.wav'):
            mimetype = 'audio/wav'
        elif filename.lower().endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.lower().endswith('.flac'):
            mimetype = 'audio/flac'
        # Add others if needed

        return send_from_directory(
            directory=abs_snippet_dir, # Use absolute path
            path=filename,
            mimetype=mimetype # Use determined MIME type
            )

    # <<< EXCEPT BLOCKS CORRECTLY FOLLOW THE TRY BLOCK >>>
    except FileNotFoundError: # Catch abort(404) from above or potential send_from_directory error
        # The abort(404) inside the try block likely handles this already,
        # but having this catch ensures we log/abort if something else raises FileNotFoundError.
        current_app.logger.warning(f"FileNotFoundError exception caught for snippet: job={job_id}, temp_id={temp_id}, filename={filename}")
        abort(404, description="Snippet not found.") # Ensure consistent 404
    except Exception as e:
        # Log the filename which was defined before the try block
        current_app.logger.error(f"Error serving snippet {filename} for job {job_id}: {e}", exc_info=True)
        abort(500, description="Error retrieving snippet file.")

# <<< NO DUPLICATED CODE BLOCK HERE >>>


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure base output directory exists on startup
    os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True) # Use config constant
    logger.info(f"Ensured output directory exists: {config.DEFAULT_OUTPUT_DIR}")
    # Set logger level based on debug flag BEFORE running
    log_level_to_set = logging.DEBUG if app.debug else logging.INFO
    logging.getLogger('flask.app').setLevel(log_level_to_set) # Set level for Flask's logger
    # You might need to adjust levels for other loggers (like werkzeug) if needed
    # logging.getLogger('werkzeug').setLevel(logging.WARNING) # Example: Quieter werkzeug logs

    logger.info(f"Starting Flask development server (Debug Mode: {app.debug})...")
    # Run with debug=True for development to see DEBUG logs and get auto-reloading
    # use_reloader=False is often needed when running with Celery workers to avoid conflicts
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) # Enable debug mode


