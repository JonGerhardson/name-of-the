# web_ui.py
# Updated to remove whisper_compute argument for openai-whisper compatibility
# Integrated whisper_prompt and central DB directory check
# Integrated whisper_language selection
# *** ADDED: Handling for diarization tuning parameters ***
import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, abort, current_app, flash, redirect # Added flash, redirect
import tasks # Import your Celery tasks module
from celery_app import celery # Import the celery instance
import config # Import config for defaults
from werkzeug.utils import secure_filename

# --- Configuration ---
# These are defaults, Flask config below is the primary source within the app context
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# --- App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['OUTPUT_FOLDER'] = config.DEFAULT_OUTPUT_DIR # Load from config
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_CHANGE_ME') # Needed for flash messages

# --- Logging Setup ---
log_format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format) # Set basicConfig level to DEBUG initially
logger = logging.getLogger('flask.app') # Use Flask's standard logger name

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
    # Pass default model and language list to template if needed for dropdowns
    # Example: You might want to define available languages in config.py
    available_languages = getattr(config, 'AVAILABLE_WHISPER_LANGUAGES', ['en', 'es', 'fr', 'de', 'auto']) # Example
    return render_template(
        'index.html',
        default_whisper_model=config.DEFAULT_WHISPER_MODEL,
        available_languages=available_languages,
        default_language='en' # Or get from config
        )

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and starts the STAGE 1 processing task."""
    current_app.logger.debug("Received request to /upload")
    if 'file' not in request.files:
        current_app.logger.warning("Upload attempt with no file part.")
        # Use flash for user feedback
        flash('No audio file part in the request.', 'error')
        # Redirect or render template with error
        return redirect(url_for('index')) # Redirect back to index
        # return jsonify({'error': 'No file part'}), 400 # Alternative API style response

    file = request.files['file']
    if file.filename == '':
        current_app.logger.warning("Upload attempt with no selected file.")
        flash('No audio file selected.', 'error')
        return redirect(url_for('index'))
        # return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(current_app.config['OUTPUT_FOLDER'], job_id)
        upload_subdir = os.path.join(job_output_dir, "upload") # Good practice

        try:
            os.makedirs(upload_subdir, exist_ok=True) # Creates parent job_output_dir too
            original_filename = secure_filename(file.filename)
            save_path = os.path.join(upload_subdir, original_filename)
            file.save(save_path)
            current_app.logger.info(f"Job {job_id}: File '{original_filename}' saved to {save_path}")
        except OSError as e:
            current_app.logger.error(f"Job {job_id}: Error creating directories or saving file: {e}", exc_info=True)
            flash('Server error creating directories or saving file.', 'error')
            return redirect(url_for('index'))
            # return jsonify({'error': 'Server error creating directories or saving file'}), 500
        except Exception as e:
            current_app.logger.error(f"Job {job_id}: Error saving uploaded file: {e}", exc_info=True)
            flash('Server error saving file.', 'error')
            return redirect(url_for('index'))
            # return jsonify({'error': 'Server error saving file'}), 500

        try:
            # --- Get standard optional arguments ---
            whisper_model = request.form.get('whisper_model', config.DEFAULT_WHISPER_MODEL)
            normalize_numbers_str = request.form.get('normalize_numbers', 'false')
            normalize_numbers = normalize_numbers_str.lower() == 'true'
            remove_fillers_str = request.form.get('remove_fillers', 'false')
            remove_fillers = remove_fillers_str.lower() == 'true'
            whisper_prompt = request.form.get('whisper_prompt', '').strip()
            whisper_language = request.form.get('whisper_language', 'en')
            if whisper_language == 'auto':
                whisper_language = None # Whisper library expects None for auto-detect

            # --- Get Output Formats ---
            output_formats = request.form.getlist('output_formats') # Gets list of checked values
            if not output_formats: # Ensure at least one format is selected
                output_formats = ['txt'] # Default to txt if none selected
                flash('No output format selected, defaulting to Standard Text (.txt).', 'warning')

            # --- Get Merge Heuristic Controls ---
            enable_merge_heuristic_str = request.form.get('enable_merge_heuristic', 'false')
            enable_merge_heuristic = enable_merge_heuristic_str.lower() == 'true'
            merge_gap_threshold_ms_str = request.form.get('merge_gap_threshold_ms', str(config.DEFAULT_MERGE_GAP_THRESHOLD_MS)).strip()
            try:
                merge_gap_threshold_ms = int(merge_gap_threshold_ms_str) if merge_gap_threshold_ms_str else config.DEFAULT_MERGE_GAP_THRESHOLD_MS
                if merge_gap_threshold_ms < 0: merge_gap_threshold_ms = config.DEFAULT_MERGE_GAP_THRESHOLD_MS
            except (ValueError, TypeError):
                 merge_gap_threshold_ms = config.DEFAULT_MERGE_GAP_THRESHOLD_MS
                 flash(f'Invalid Merge Gap Threshold value, using default: {merge_gap_threshold_ms}ms.', 'warning')

            # --- START: Retrieve and Validate Pyannote Parameters ---
            # Provide sensible defaults as strings first
            default_min_duration = "0.05" # Example default - adjust as needed
            default_cluster_threshold = "0.8" # Example default - adjust as needed

            min_duration_on_str = request.form.get('min_duration_on', '').strip() # Default to empty string if not present
            clustering_threshold_str = request.form.get('clustering_threshold', '').strip()

            # Use default if the input is empty or invalid
            min_duration_on = float(default_min_duration) # Start with default
            if min_duration_on_str: # Only try to parse if user provided something
                try:
                    parsed_min_duration = float(min_duration_on_str)
                    if 0 <= parsed_min_duration <= 1.0: # Basic sanity check range
                        min_duration_on = parsed_min_duration
                    else:
                        flash(f'Min Segment Duration ({parsed_min_duration}s) out of range [0, 1], using default: {min_duration_on}s.', 'warning')
                except (ValueError, TypeError):
                    flash(f'Invalid Min Segment Duration value ("{min_duration_on_str}"), using default: {min_duration_on}s.', 'warning')

            clustering_threshold = float(default_cluster_threshold) # Start with default
            if clustering_threshold_str: # Only try to parse if user provided something
                try:
                    parsed_cluster_threshold = float(clustering_threshold_str)
                    # Allow slightly wider range as optimal value might be > 1 sometimes
                    if 0.1 <= parsed_cluster_threshold <= 1.5:
                        clustering_threshold = parsed_cluster_threshold
                    else:
                         flash(f'Clustering Threshold ({parsed_cluster_threshold}) out of range [0.1, 1.5], using default: {clustering_threshold}.', 'warning')
                except (ValueError, TypeError):
                    flash(f'Invalid Clustering Threshold value ("{clustering_threshold_str}"), using default: {clustering_threshold}.', 'warning')

            # Create the dictionary for pipeline kwargs
            diarization_params = {
                "segmentation.min_duration_on": min_duration_on,
                "clustering.threshold": clustering_threshold
            }
            current_app.logger.debug(f"Job {job_id}: Using Diarization Params: {diarization_params}")
            # --- END: Retrieve and Validate Pyannote Parameters ---


            # Collect all relevant pipeline kwargs for the task
            task_kwargs = {
                'whisper_model': whisper_model,
                'embedding_model_name': config.DEFAULT_EMBEDDING_MODEL,
                'punctuation_model_name': config.DEFAULT_PUNCTUATION_MODEL,
                'similarity_threshold': config.DEFAULT_SIMILARITY_THRESHOLD,
                'min_segment_duration': config.DEFAULT_MIN_SEGMENT_DURATION,
                'punctuation_chunk_size': config.DEFAULT_PUNCTUATION_CHUNK_SIZE,
                'normalize_numbers': normalize_numbers,
                'remove_fillers': remove_fillers,
                'filler_words': config.DEFAULT_FILLER_WORDS,
                'processing_device_requested': config.DEFAULT_PROCESSING_DEVICE,
                'whisper_device_requested': config.DEFAULT_WHISPER_DEVICE,
                'whisper_prompt': whisper_prompt,
                'whisper_language': whisper_language,
                'output_formats': output_formats, # Pass selected formats
                'enable_merge_heuristic': enable_merge_heuristic,
                'merge_gap_threshold_ms': merge_gap_threshold_ms,
                # --- Pass the new dictionary ---
                'diarization_kwargs': diarization_params
            }
            current_app.logger.debug(f"Job {job_id}: Prepared final task_kwargs: {task_kwargs}")

            # Pass the *saved path* of the uploaded file
            task = tasks.run_identification_stage.delay(
                input_audio_path=save_path,
                output_dir=job_output_dir,
                **task_kwargs
            )
            current_app.logger.info(f"Job {job_id}: Started Stage 1 Celery task {task.id}")
            # Return task ID and job ID for the frontend to track
            return jsonify({'task_id': task.id, 'job_id': job_id}), 202 # Accepted

        except Exception as e:
            current_app.logger.error(f"Job {job_id}: Error during task start preparation: {e}", exc_info=True)
            flash('Server error preparing processing task.', 'error')
            return redirect(url_for('index'))
            # return jsonify({'error': 'Server error during task start'}), 500
    else:
        current_app.logger.warning(f"Upload attempt with disallowed file type: {file.filename}")
        flash(f'File type not allowed: {file.filename}. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
        return redirect(url_for('index'))
        # return jsonify({'error': 'File type not allowed'}), 400


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

        # State Handling (remains unchanged from previous version)
        if task.state == 'PENDING':
            response['status'] = 'Pending...'
        elif task.state == 'STARTED':
            response['status'] = task_meta.get('status', 'Started...')
        elif task.state == 'PROGRESS':
            response['status'] = task_meta.get('status', 'Processing...')
            response['progress'] = task_meta.get('progress', 0)
        elif task.state == 'AWAITING_ENROLLMENT':
            response['status'] = task_meta.get('message', 'Enrollment Required')
            response['result'] = task_meta # Pass meta for UI
            if 'unknown_speakers' not in response['result']:
                 current_app.logger.warning(f"Task {task_id} AWAITING_ENROLLMENT but 'unknown_speakers' missing in meta.")
        elif task.state == 'SUCCESS':
            response['result'] = task_return_value
            response['progress'] = 100
            response['status'] = 'Complete' # Default
            if isinstance(task_return_value, dict):
                response['status'] = task_return_value.get('message', response['status'])
            elif task_return_value is None:
                 current_app.logger.warning(f"Task {task_id} succeeded but returned None result.")
        elif task.state == 'FAILURE':
            error_info = str(task.info) if task.info else "Unknown error"
            response['status'] = task_meta.get('status', 'Failed') # Use meta status if available
            response['error'] = task_meta.get('error', error_info) # Prefer error from meta if possible
            response['result'] = {'error': response['error']} # Store error in result for JS
            current_app.logger.error(f"Task {task_id} failed: {response['error']}", exc_info=isinstance(task.info, Exception))
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
        original_kwargs = data.get('original_kwargs') # This will include prompt & language if sent initially
        current_app.logger.debug(f"Enrollment data received: job_id={job_id}, output_dir={output_dir}, map_keys={list(enrollment_map.keys()) if enrollment_map else None}, original_kwargs_keys={list(original_kwargs.keys()) if original_kwargs else None}")

        # Validate required data
        missing = []
        if not job_id: missing.append('job_id')
        if not output_dir: missing.append('output_dir')
        if enrollment_map is None: missing.append('enrollment_map') # Allow empty map {}
        if original_kwargs is None: missing.append('original_kwargs') # Allow empty dict {}

        if missing:
            current_app.logger.warning(f"Enrollment request missing data. Missing: {missing}")
            return jsonify({'error': f'Missing required data for enrollment: {", ".join(missing)}'}), 400

        # Security Check: Ensure output_dir is within the main output folder
        abs_output_folder = os.path.abspath(current_app.config['OUTPUT_FOLDER'])
        abs_output_dir = os.path.abspath(output_dir)

        # Check if the resolved path starts with the output folder base path
        # Also check if the last component matches the job_id for extra safety
        if not abs_output_dir.startswith(abs_output_folder) or os.path.basename(abs_output_dir) != job_id:
            current_app.logger.error(f"Potential path manipulation attempt or mismatch for job {job_id}, output_dir {output_dir}")
            return jsonify({'error': 'Invalid output directory or job ID mismatch'}), 400

        current_app.logger.info(f"Job {job_id}: Received enrollment request with map: {enrollment_map}")

        # Start Stage 2 Task (passing original_kwargs which includes prompt/lang, etc.)
        finalize_task = tasks.run_finalization_stage.delay(
            output_dir=output_dir,
            enrollment_map=enrollment_map,
            original_kwargs=original_kwargs # Pass the whole dict
        )
        current_app.logger.info(f"Job {job_id}: Started Stage 2 (Finalization) Celery task {finalize_task.id}")
        return jsonify({'finalize_task_id': finalize_task.id}), 202 # Accepted

    except Exception as e:
        current_app.logger.error(f"Error processing enrollment request: {e}", exc_info=True)
        return jsonify({'error': 'Server error processing enrollment'}), 500


@app.route('/results/<job_id>/<path:filename>')
def get_result_file(job_id, filename):
    """Serves a specific result file associated with a completed job."""
    current_app.logger.debug(f"Result file request: job_id={job_id}, filename={filename}")
    secure_job_id = secure_filename(job_id)

    # Path Checks (remains unchanged)
    if secure_job_id != job_id:
        current_app.logger.warning(f"Invalid characters detected in job ID {job_id}")
        abort(400, description="Invalid job ID format.")
    if '..' in filename or filename.startswith('/'):
        current_app.logger.warning(f"Path traversal attempt detected for job {job_id}, filename {filename}")
        abort(400, description="Invalid filename path.")

    try:
        job_directory = os.path.join(current_app.config['OUTPUT_FOLDER'], secure_job_id)
        abs_output_folder = os.path.abspath(current_app.config['OUTPUT_FOLDER'])
        abs_job_directory = os.path.abspath(job_directory)

        if not abs_job_directory.startswith(abs_output_folder):
            current_app.logger.error(f"Attempt to access directory outside designated output folder: {job_directory}")
            abort(404, description="Result file not found (invalid path).")

        full_file_path = os.path.join(abs_job_directory, filename)
        current_app.logger.info(f"Attempting to serve file: '{filename}' from directory: '{abs_job_directory}'")
        current_app.logger.debug(f"Checking existence of full path: '{full_file_path}'")

        if not os.path.isfile(full_file_path):
            current_app.logger.warning(f"Result file does not exist at path: {full_file_path}")
            abort(404, description="Result file not found on server.")

        # Serve the file
        return send_from_directory(
            directory=abs_job_directory,
            path=filename,
            as_attachment=False # Allow viewing in browser for text/json
        )

    except FileNotFoundError:
        log_path = os.path.join(current_app.config.get('OUTPUT_FOLDER', ''), secure_job_id, filename or '')
        current_app.logger.warning(f"send_from_directory failed for: {filename} in job {job_id} (Path: {log_path})")
        abort(404, description="Result file not found.")
    except Exception as e:
        current_app.logger.error(f"Error serving result file for job {job_id}, filename {filename}: {e}", exc_info=True)
        abort(500, description="Error retrieving result file.")


@app.route('/snippets/<job_id>/<temp_id>')
def get_speaker_snippet(job_id, temp_id):
    """Serves a specific speaker audio snippet file."""
    current_app.logger.debug(f"Snippet request received: job_id={job_id}, temp_id={temp_id}")

    # Basic Security (remains unchanged)
    secure_job_id = secure_filename(job_id)
    # Secure temp_id as well - prevent directory traversal using temp_id
    secure_temp_id = secure_filename(temp_id)
    if secure_temp_id != temp_id or '..' in temp_id or temp_id.startswith('/'):
         current_app.logger.warning(f"Path traversal or invalid characters attempt detected in temp_id: {temp_id}")
         abort(400, description="Invalid speaker ID path.")
    if secure_job_id != job_id:
        current_app.logger.warning(f"Invalid characters detected in snippet request job_id: {job_id}")
        abort(400, description="Invalid job ID format.")

    # Construct filename using secured temp_id
    filename = f"snippet_{secure_temp_id}.wav"
    current_app.logger.debug(f"Constructed snippet filename: '{filename}'")

    try:
        base_output_folder = current_app.config['OUTPUT_FOLDER']
        snippet_subdir_name = getattr(config, 'SNIPPETS_SUBDIR', 'snippets') # Use config or default
        snippet_dir = os.path.join(base_output_folder, secure_job_id, snippet_subdir_name)

        # Security Check (remains unchanged)
        abs_output_folder = os.path.abspath(base_output_folder)
        abs_snippet_dir = os.path.abspath(snippet_dir)

        if not abs_snippet_dir.startswith(abs_output_folder):
            current_app.logger.error(f"Attempt to access snippets directory outside designated output folder: {snippet_dir}")
            abort(404, description="Snippet not found (invalid path).")

        # Log and Check Path (remains unchanged)
        full_file_path = os.path.join(abs_snippet_dir, filename)
        current_app.logger.info(f"Attempting to serve snippet: '{filename}' from directory: '{abs_snippet_dir}'")
        current_app.logger.debug(f"Checking existence of full path: '{full_file_path}'")

        if not os.path.isfile(full_file_path):
            current_app.logger.warning(f"Snippet file does not exist at path: {full_file_path}")
            abort(404, description="Snippet file not found on server.")
        else:
            current_app.logger.info(f"Snippet file FOUND at path: {full_file_path}. Proceeding to serve.")

        # Serve the file (remains unchanged)
        mimetype = 'audio/wav' # Assuming wav snippets
        return send_from_directory(
            directory=abs_snippet_dir,
            path=filename,
            mimetype=mimetype
        )

    except FileNotFoundError:
        current_app.logger.warning(f"FileNotFoundError exception caught for snippet: job={job_id}, temp_id={temp_id}, filename={filename}")
        abort(404, description="Snippet not found.")
    except Exception as e:
        current_app.logger.error(f"Error serving snippet {filename} for job {job_id}: {e}", exc_info=True)
        abort(500, description="Error retrieving snippet file.")


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure directories exist on startup (remains unchanged)
    try:
        os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
        logger.info(f"Ensured output directory exists: {config.DEFAULT_OUTPUT_DIR}")
        os.makedirs(config.CENTRAL_DB_DIR, exist_ok=True)
        logger.info(f"Ensured central speaker database directory exists: {config.CENTRAL_DB_DIR}")
    except OSError as e:
         logger.error(f"Could not create required directories on startup: {e}", exc_info=True)

    # Set logger level based on debug flag BEFORE running
    log_level_to_set = logging.DEBUG if app.debug else logging.INFO
    logging.getLogger('flask.app').setLevel(log_level_to_set)

    logger.info(f"Starting Flask development server (Debug Mode: {app.debug})...")
    # use_reloader=False is often recommended when running with Celery
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

