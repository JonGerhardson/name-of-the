# Name of the . . . 

## A self-hosted transcription app with speaker identification 

This project provides a pipeline to process audio files. It performs speech-to-text transcription, identifies different speakers, matches speakers against a database, and allows adding new speakers to the database. The output text can be formatted with punctuation and normalized numbers.

It includes a web interface for uploading files and managing speaker enrollment, and a command-line tool for direct processing. It uses faster-whisper for transcription, pyannote.audio for diarization, and speech-brains for speaker embeddings. This app basically just glues them all together with a UI. 


## Updates 5-5-2025

Now uses docker-compose, defaults to running on GPU, remembers speaker IDs across different transcripts, allows for whisper prompting, and some other things I'm forgetting right now. Need to update this readme. To run: 

```
git clone https://github.com/JonGerhardson/name-of-the.git
```
```
cd name-of-the

docker compose build 

docker compose up -d

```

Web app will be running at http://localhost:5000


## Setup & Usage

This section describes how to set up and run the application using Docker (recommended) or manually.

### Using Docker

Docker provides a containerized environment with all dependencies included.

1.  **Prerequisites:**
    * Install Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
    * Install Git: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
    * **Hugging Face Account & Token:** You need a Hugging Face account ([https://huggingface.co/join](https://huggingface.co/join)) and an access token ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) with `read` permissions.
    * **Accept Model User Conditions:** You *must* accept the user conditions for the following models on the Hugging Face Hub before using them:
        * `pyannote/speaker-diarization-3.1` ([https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1))
        * `pyannote/segmentation-3.0` ([https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)) (Dependency of diarization)
    * Install NVIDIA Container Toolkit (if using GPU): [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Ensure `faiss-gpu` is used in `requirements.txt` if needed.

(Note: Hugging Face token is only needed to download the pyannote libraries, no transcription data is sent anywhere.) 

2.  **Clone the Repository:**
    Open a terminal and clone the project code:
    ```bash
    git clone [https://github.com/JonGerhardson/name-of-the.git](https://github.com/JonGerhardson/name-of-the.git)
    cd name-of-the
    ```

3.  **Prepare Hugging Face Token File (Optional but Recommended):**
    * Create a file named `hf-token.txt` in the cloned `name-of-the` directory.
    * Paste your Hugging Face access token into this file and save it. This file will be used by the Docker container.

4.  **Build Docker Image:**
    From within the cloned project directory (`name-of-the`), run:
    ```bash
    docker build -t name-of-the-app .
    ```
    *(Using a Docker image tag like `name-of-the-app`)*

5.  **Prepare Host Directories:**
    Create directories on your computer to store output transcripts, logs, and the speaker database persistently:
    ```bash
    # mkdir -p ./host_audio_input # Input is handled per-job inside output dir
    mkdir -p ./host_transcripts_output
    ```
    *(Note: The application saves job-specific inputs and all outputs within subdirectories of the main output folder)*

6.  **Run Docker Container:**
    ```bash
    docker run -d --name name-of-the-instance \
      -p 5000:5000 \
      # Mount the main output directory
      -v $(pwd)/host_transcripts_output:/app/transcripts_output \
      # Mount the HF token file (if you created it in step 3)
      -v $(pwd)/hf-token.txt:/app/hf-token.txt \
      # Optionally set Celery broker/backend URLs if not using default Redis inside container
      # -e CELERY_BROKER_URL=redis://your-redis-host:6379/0 \
      # -e CELERY_RESULT_BACKEND=redis://your-redis-host:6379/0 \
      # Optionally set Flask Secret Key
      # -e FLASK_SECRET_KEY='your_super_secret_key' \
      # Add --gpus all if using GPU acceleration
      name-of-the-app
    ```
    *(Using a container name like `name-of-the-instance`)*
    * The `entrypoint.sh` script (assumed) starts Redis (if included in image), the Celery worker, and the Gunicorn/Flask web server. The application will look for the token at `/app/hf-token.txt`.

### Manual Installation (Requires Python 3.x)

Use this method if you prefer not to use Docker.

1.  **Prerequisites:**
    * Install `ffmpeg`: Instructions vary by operating system (e.g., `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).
    * Install and run `redis-server`: (e.g., `sudo apt install redis-server` or use a managed service). Ensure the Redis server is running.
    * Install Git: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
    * **Hugging Face Account & Token:** You need a Hugging Face account ([https://huggingface.co/join](https://huggingface.co/join)) and an access token ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) with `read` permissions.
    * **Accept Model User Conditions:** You *must* accept the user conditions for the following models on the Hugging Face Hub before using them:
        * `pyannote/speaker-diarization-3.1` ([https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1))
        * `pyannote/segmentation-3.0` ([https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)) (Dependency of diarization)

2.  **Clone Repository:**
    ```bash
    git clone [https://github.com/JonGerhardson/name-of-the.git](https://github.com/JonGerhardson/name-of-the.git)
    cd name-of-the
    ```

3.  **Set up Python Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install gunicorn  # Included in Dockerfile, install if needed manually
    ```

4.  **Prepare Configuration (Optional but Recommended):**
    * Create a file named `hf-token.txt` in the project root (`name-of-the` directory).
    * Paste your Hugging Face access token into this file and save it. The application will look for this file by default (path configured in `config.py`).

5.  **Start Services:**
    Open three separate terminals in the project directory with the virtual environment activated.
    * Terminal 1: Start Redis (if installed locally and not already running).
        ```bash
        redis-server
        ```
    * Terminal 2: Start the Celery background worker.
        ```bash
        # Ensure Redis is running and accessible via default URL or env vars
        celery -A celery_app.celery worker --loglevel=info
        ```
    * Terminal 3: Start the Web Server.
        * For production:
            ```bash
            # Set FLASK_SECRET_KEY environment variable for production
            # export FLASK_SECRET_KEY='your_super_secret_key'
            gunicorn --bind 0.0.0.0:5000 web_ui:app
            ```
        * For development:
            ```bash
            python web_ui.py
            ```

### Using the Application

1.  **Access Web Interface:**
    Open a web browser to `http://localhost:5000` (or the server's IP address).

2.  **Process Audio:**
    * Upload an audio file (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`).
    * Select processing options (Whisper model, normalization).
    * Submit the file. A unique job ID is created, and processing starts in the background (`tasks.run_identification_stage`).

3.  **Monitor Progress:**
    The web UI polls the `/status/<task_id>` endpoint to show progress.

4.  **Enroll Speakers (If prompted):**
    * If unknown speakers are detected (status `AWAITING_ENROLLMENT`), the UI lists them.
    * Audio snippets (`.wav`) are generated (`audio_processing.save_speaker_snippet`) and served via `/snippets/<job_id>/<temp_id>`.
    * Text previews are extracted from the intermediate transcript.
    * Enter a name for each speaker to enroll and submit via the `/enroll` endpoint. This triggers the finalization task (`tasks.run_finalization_stage`).
    * 
    


https://github.com/user-attachments/assets/70c819c3-537c-4c0d-8530-a31bc83bbf56





5.  **Download Results:**
    Once processing is complete (status `SUCCESS`), download links for the final transcript files (`final_normalized_transcript.txt`, `intermediate_identified_transcript.json`, `processing_log.log`) appear, served via `/results/<job_id>/<filename>`.
    * **Note:** If download links are unavailable or not working, you may need to retrieve the files directly from the output directory on your host machine, typically located at `./host_transcripts_output/<job_id>/` (when using the recommended Docker setup), where `<job_id>` is the unique identifier for your processing job.

6.  **Using the Command-Line Tool (`pipeline.py`):**
    * Activate the Python virtual environment (`source venv/bin/activate`).
    * Run the main pipeline script:
        ```bash
        python pipeline.py <path_to_your_audio_file> --output-dir ./cli_output [options]
        ```
    * Use `python pipeline.py --help` to see all available options matching `config.py` defaults.
    * The tool performs the full pipeline in one go and may prompt for speaker names interactively in the terminal during enrollment (`speaker_id.enroll_new_speakers_cli`).

## Features

* **Audio Transcription:** Converts speech to text using `faster-whisper` (`models.load_whisper_model`).
* **Speaker Diarization:** Determines speaker segments using `pyannote.audio` (`models.load_diarization_pipeline`). Requires HF token with accepted user conditions.
* **Speaker Identification:** Matches speakers against known voice profiles using embeddings (`speechbrain/spkrec-ecapa-voxceleb` via `models.load_embedding_model`, `speaker_id.get_speaker_embeddings`) and FAISS similarity search (`persistence.load_or_create_faiss_index`, `speaker_id.identify_speakers`).
* **Speaker Enrollment:** Adds new speakers (name and embedding) to the database (FAISS index, JSON map) via CLI prompts (`speaker_id.enroll_new_speakers_cli`) or programmatically based on Web UI input (`speaker_id.enroll_speakers_programmatic`).
* **Text Post-Processing (`text_processing.py`):**
    * Applies punctuation using a token classification model (`models.load_punctuation_model`).
    * Normalizes numbers to words (optional, using `inflect`).
    * Removes filler words (optional, using regex and `config.DEFAULT_FILLER_WORDS`).
* **Web Interface (`web_ui.py`):** Flask UI for file uploads, option selection, task monitoring, interactive enrollment (with audio snippets and text preview), and result downloads.
* **Asynchronous Task Processing (`tasks.py`, `celery_app.py`):** Uses Celery with Redis for handling two main background task stages: identification and finalization.
* **Audio Handling (`audio_processing.py`):** Loads, resamples, converts to mono, extracts segments per speaker, and generates short `.wav` snippets for enrollment UI.
* **Configuration (`config.py`):** Centralized default settings for models, paths, parameters.
* **Persistence (`persistence.py`):** Manages saving and loading of FAISS index, speaker map, and intermediate/final transcript files.
* **Logging (`log_setup.py`):** Configurable logging to console and file (`processing_log.log` within each job's output directory).
* **Docker Support (`Dockerfile`):** Packages application, dependencies (Python & system like ffmpeg, redis), and sets up the execution environment.

## Architecture / Components

* **`pipeline.py`:** Main pipeline orchestrator and CLI entry point.
* **`web_ui.py`:** Flask web application.
* **`tasks.py`:** Defines Celery tasks (`run_identification_stage`, `run_finalization_stage`).
* **`celery_app.py`:** Celery application configuration.
* **`config.py`:** Default configuration values.
* **`models.py`:** Functions for loading ML models (ASR, Diarization, Embedding, Punctuation).
* **`speaker_id.py`:** Logic for speaker embedding generation, identification, and enrollment.
* **`audio_processing.py`:** Audio loading, resampling, segmenting, snippet generation.
* **`text_processing.py`:** Punctuation restoration and text normalization.
* **`persistence.py`:** Saving/loading of speaker database (FAISS, JSON map) and transcripts.
* **`utils.py`:** Helper functions (device checks, cleanup, token loading).
* **`log_setup.py`:** Logging configuration function.
* **`requirements.txt`:** Python dependencies.
* **`Dockerfile`:** Docker image definition.
* **`entrypoint.sh`** (*Assumed*): Container startup script.
* **`templates/`** (*Assumed*): HTML templates for Flask (e.g., `index.html`).
* **`static/`** (*Assumed*): CSS/JavaScript files for Flask.

## Configuration

Configuration is managed through defaults in `config.py`, command-line arguments (`pipeline.py`), web UI form options, and environment variables.

* **`config.py`:** Defines defaults for:
    * File paths/names (Output base directory, log file, speaker DB files, intermediate/final transcripts, **HF token file**).
    * Model names (Whisper, **Diarization (requires token)**, Embedding, Punctuation).
    * Processing parameters (Devices, Similarity threshold, Min segment duration, Punctuation chunk size, Embedding dimension).
    * Normalization options (Enable flags, Filler word list).
    * Logging levels.
* **CLI Arguments (`pipeline.py --help`):** Allows overriding most defaults from `config.py` for command-line execution.
* **Web UI Options:** Allows overriding Whisper model, number normalization, and filler word removal flags during upload.
* **Environment Variables:**
    * `CELERY_BROKER_URL`: URL for the Celery message broker (Default: `redis://localhost:6379/0`).
    * `CELERY_RESULT_BACKEND`: URL for the Celery result backend (Default: `redis://localhost:6379/0`).
    * `FLASK_SECRET_KEY`: Secret key for Flask session security (important for production).

## Speaker Database

The application maintains a persistent speaker database within the configured output directory structure (specifically, relative to where `faiss_file` and `map_file` paths resolve, often the base output directory).

* **`speaker_embeddings.index`:** A FAISS (IndexFlatIP) file storing L2-normalized speaker voice embeddings (currently default dimension: 192).
* **`speaker_names.json`:** A JSON file mapping the internal FAISS index ID (integer) to the user-provided speaker name (string).

These files are loaded at the start of identification and updated upon successful enrollment. Back up these files regularly if the enrolled speakers are important. Manual editing of these files is possible but requires caution to maintain consistency between the index and the map.

---
*README generated based on provided project files.*


