web_ui.py: This script sets up a Flask web application. It serves the HTML interface, handles audio file uploads, initiates the processing by calling Celery tasks defined in tasks.py, monitors the status of these tasks, handles the user input for enrolling unknown speakers, and provides endpoints to download the final transcript files and listen to audio snippets of unknown speakers for identification purposes.
celery_app.py: This configures Celery, a distributed task queue. It sets up Redis as the message broker (to send tasks) and the result backend (to store task results). It tells Celery to look for task definitions within the tasks.py module.
tasks.py: Defines the background tasks that the web UI triggers.

    run_identification_stage: This task takes the uploaded audio path and runs the first part of the pipeline (pipeline.run_full_pipeline in "identify_only" mode). It handles ASR, diarization, and initial speaker identification. If unknown speakers are found, it updates its state to AWAITING_ENROLLMENT and returns information (like temporary IDs, timestamps, and audio snippet paths) needed by the web UI for the user to enroll them.
    run_finalization_stage: This task is triggered after the user submits names for unknown speakers. It takes the enrollment information and runs the second part of the pipeline (pipeline.run_full_pipeline in "finalize_enrollment" mode), which updates the speaker database, applies the final speaker names to the transcript, performs text post-processing (punctuation, normalization), and saves the final results.

pipeline.py: This is the main orchestrator of the processing workflow.

    run_full_pipeline: This core function controls the different processing stages based on the enrollment_stage argument. It calls functions from other modules to load models, run ASR (using openai-whisper), run diarization, combine results, perform speaker identification (calling speaker_id.run_speaker_identification), handle enrollment (CLI or programmatic via speaker_id), run post-processing (punctuation and normalization via text_processing), and save intermediate and final files using persistence.py. It also handles temporary file management between stages.

models.py: This module is responsible for loading and initializing the various machine learning models used throughout the pipeline.

    load_whisper_model: Loads the openai-whisper model for automatic speech recognition (ASR).
    load_diarization_pipeline: Loads the pyannote.audio pipeline for speaker diarization (determining who speaks when).
    load_embedding_model: Loads a SpeechBrain model to generate numerical representations (embeddings) of speaker voices.
    load_punctuation_model: Loads a Hugging Face transformers pipeline for adding punctuation to the transcribed text.

audio_processing.py: Contains utility functions for handling audio data.

    load_audio: Loads an audio file, resamples it to a standard rate (16kHz), and converts it to mono.
    extract_speaker_audio_segments: Extracts chunks of audio corresponding to each speaker based on the diarization results and word timings, concatenating them and ensuring they meet a minimum duration requirement.
    save_speaker_snippet: Extracts and saves a short .wav snippet of audio for a specific speaker, used for the enrollment process in the UI.

speaker_id.py: Focuses on identifying speakers and managing the speaker database.

    get_speaker_embeddings: Generates voice embeddings from the audio segments using the loaded SpeechBrain model.
    identify_speakers: Compares the generated embeddings against a database (FAISS index) of known speakers. It assigns known speaker names or temporary IDs (like 'UNKNOWN_1') if no match above a similarity threshold is found. It returns information about new speakers, including their embeddings and timestamps.
    update_transcript_speakers: Updates the speaker labels in the transcript data based on the results of the identification step.
    run_speaker_identification: Wraps the core speaker ID logic, including extracting segments, getting embeddings, identifying speakers, saving snippets for unknown speakers (audio_processing.save_speaker_snippet), and returning the updated transcript along with information about any new speakers found.
    enroll_new_speakers_cli, enroll_speakers_programmatic: Functions to add new speakers (name and embedding) to the FAISS database and the speaker map file, either interactively via command line or programmatically based on input from the web UI.

text_processing.py: Handles the final cleaning and formatting of the transcribed text.

    format_punctuated_output: Takes the raw output from the punctuation model (which classifies tokens) and reconstructs readable text with correct punctuation and capitalization.
    apply_punctuation: Applies the punctuation model to the speaker turns generated earlier in the pipeline.
    normalize_text_custom: Performs text normalization, optionally converting numbers to words (using inflect) and removing filler words like "um" or "uh" (using predefined lists and potentially cucco for whitespace cleanup).

persistence.py: Manages reading and writing data to disk. It includes functions to:

    Load and save the FAISS index (efficient vector similarity search).
    Load and save the speaker map (mapping FAISS index IDs to speaker names in JSON).
    Save and load intermediate transcript data (JSON format).
    Save the final, formatted transcript as a text file with speaker labels.
