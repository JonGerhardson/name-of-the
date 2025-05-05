# speaker_id.py (Save Snippets)
import torch
import numpy as np
import faiss
import logging
import traceback
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

# Import necessary modules
import persistence # Import the whole module
import audio_processing # Import the audio processing functions

# Assuming persistence functions like save_faiss_index, save_speaker_map are used elsewhere or imported specifically
# from persistence import save_faiss_index, save_speaker_map

logger = logging.getLogger(__name__)

# --- Constants ---
SNIPPET_MAX_DURATION_SEC = 5.0 # Max length of audio snippet for enrollment UI
SNIPPETS_SUBDIR = "snippets" # Subdirectory within job output dir to store snippets

# --- get_speaker_embeddings (Keep as is) ---
def get_speaker_embeddings(speaker_audio_segments: Dict[str, torch.Tensor], embedding_model: Any, device: str) -> Dict[str, np.ndarray]:
    """Generates speaker embeddings using the provided SpeechBrain model."""
    embeddings = {}
    logger.info("Generating speaker embeddings...")

    if not speaker_audio_segments:
        logger.warning("No speaker audio segments provided for embedding generation.")
        return embeddings
    if not embedding_model:
        logger.error("Embedding model not loaded. Cannot generate embeddings.")
        return embeddings # Return empty dict if model is missing

    try:
        model_device = torch.device(device)
        embedding_model.to(model_device)
        embedding_model.eval()

        with torch.no_grad():
            for speaker, audio_tensor in speaker_audio_segments.items():
                logger.info(f"  - Processing {speaker}...")
                try:
                    # Ensure tensor is on the correct device and has batch dim
                    audio_tensor = audio_tensor.to(model_device)
                    # Standardize input shape (assuming model expects [batch, samples])
                    if audio_tensor.dim() == 3 and audio_tensor.shape[1] == 1: # [1, 1, samples] -> [1, samples]
                        audio_tensor = audio_tensor.squeeze(1)
                    elif audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1: # [channels>1, samples] -> [1, samples] (take mean)
                        logger.debug(f"     Input tensor has shape {audio_tensor.shape}. Taking mean across channels.")
                        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                    elif audio_tensor.dim() == 1: # [samples] -> [1, samples]
                        audio_tensor = audio_tensor.unsqueeze(0)
                    elif audio_tensor.dim() != 2 or audio_tensor.shape[0] != 1: # Check if it's already [1, samples] or unexpected
                         logger.warning(f"     Unexpected audio tensor shape for {speaker}: {audio_tensor.shape}. Attempting to proceed.")

                    # Generate embedding
                    embedding = embedding_model.encode_batch(audio_tensor)

                    # Handle potential multiple embeddings per segment (average if needed)
                    if embedding.dim() == 3: # e.g., [1, N, embed_dim]
                        logger.debug(f"     Averaging {embedding.shape[1]} frame-level embeddings for {speaker}.")
                        embedding = torch.mean(embedding, dim=1) # Average over N -> [1, embed_dim]

                    embedding = embedding.squeeze(0) # -> [embed_dim]

                    # Normalize the embedding (L2 norm)
                    norm = torch.linalg.norm(embedding, ord=2)
                    if norm > 1e-6: # Avoid division by zero
                        embedding = embedding / norm
                    else:
                        logger.warning(f"     Embedding norm for {speaker} is near zero ({norm}). Skipping normalization.")

                    embeddings[speaker] = embedding.cpu().numpy() # Store as numpy array
                    logger.info(f"    Generated embedding for {speaker}.")

                except Exception as e:
                    logger.error(f"     Error generating embedding for {speaker}: {e}", exc_info=True)

    except Exception as model_e:
        logger.error(f"Error during embedding generation setup or main loop: {model_e}", exc_info=True)

    logger.info(f"Embedding generation complete. Generated {len(embeddings)} embeddings.")
    return embeddings


# --- identify_speakers (Keep as is, it returns new_speaker_embeddings_info) ---
def identify_speakers(embeddings: Dict[str, np.ndarray], faiss_index: Any, speaker_map: Dict[int, str], similarity_threshold: float) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Identifies speakers using FAISS search and returns assignments and info for new speakers.
    The returned info includes the embedding needed for enrollment.
    """
    speaker_assignments = {} # Maps original label (e.g., SPEAKER_01) to identified name or temp ID
    new_speaker_embeddings_info = [] # List of {'temp_id': ..., 'embedding': ..., 'original_label': ...} for enrollment
    unknown_speaker_count = 0

    if not embeddings:
        logger.warning("No embeddings provided for identification.")
        return {}, []

    if faiss_index is None or speaker_map is None:
        logger.warning("FAISS index or speaker map not available for identification. Marking all as unknown.")
        # Mark all speakers from embeddings as unknown if DB is missing
        for speaker, embedding in embeddings.items():
            unknown_speaker_count += 1
            temp_id = f"UNKNOWN_{unknown_speaker_count}"
            speaker_assignments[speaker] = temp_id
            if isinstance(embedding, np.ndarray):
                new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
            else:
                logger.warning(f"Invalid embedding found for speaker {speaker} while marking as unknown.")
        return speaker_assignments, new_speaker_embeddings_info

    index_size = faiss_index.ntotal
    logger.info(f"Starting speaker identification for {len(embeddings)} embeddings. Index size: {index_size}.")

    if index_size == 0:
        logger.warning("FAISS index is empty. Marking all speakers as new/unknown.")
        for speaker, embedding in embeddings.items():
            unknown_speaker_count += 1
            temp_id = f"UNKNOWN_{unknown_speaker_count}"
            speaker_assignments[speaker] = temp_id
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
            else:
                logger.warning(f"Invalid embedding found for speaker {speaker} while marking as unknown (empty index). Dim: {embedding.ndim if isinstance(embedding, np.ndarray) else 'N/A'}")

    else:
        # Prepare embeddings for batch search
        query_embeddings_list = []
        query_speakers_list = []
        valid_embeddings_count = 0
        for speaker, embedding in embeddings.items():
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                query_embeddings_list.append(embedding.astype(np.float32))
                query_speakers_list.append(speaker)
                valid_embeddings_count += 1
            else:
                logger.warning(f"Invalid embedding type or dimension for {speaker}. Type: {type(embedding)}, Dim: {embedding.ndim if isinstance(embedding, np.ndarray) else 'N/A'}. Skipping.")
                speaker_assignments[speaker] = f"INVALID_EMBEDDING_{speaker}"


        if not query_embeddings_list:
            logger.warning("No valid embeddings found to query FAISS index.")
            # Return existing assignments (potentially just invalid ones) and empty new speaker list
            return speaker_assignments, new_speaker_embeddings_info

        query_embeddings_np = np.array(query_embeddings_list)
        logger.info(f"Querying FAISS with {valid_embeddings_count} valid embeddings...")

        try:
            # Search for the top 1 match using Inner Product (IP). Higher score = more similar.
            # FAISS returns distances (for L2) or negative IP scores. We need similarity.
            # For IndexFlatIP, the returned 'distances' are negative dot products.
            # Since embeddings are normalized, dot product = cosine similarity.
            # So, similarity = -distance.
            distances, indices = faiss_index.search(query_embeddings_np, k=1)

            for i, speaker in enumerate(query_speakers_list):
                # Check if search returned a valid index for this query
                if indices.size == 0 or indices[i, 0] < 0:
                    logger.info(f"  - Speaker {speaker}: FAISS search returned no valid match index.")
                    unknown_speaker_count += 1
                    temp_id = f"UNKNOWN_{unknown_speaker_count}"
                    speaker_assignments[speaker] = temp_id
                    new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': query_embeddings_np[i], 'original_label': speaker})
                    continue

                faiss_id = int(indices[i, 0])
                neg_dot_product = distances[i, 0]
                similarity = -neg_dot_product # Convert to cosine similarity

                # Ensure similarity is within expected range [-1, 1] (or [0, 1] if embeddings are non-negative)
                # Clamp similarity just in case of floating point issues
                similarity = max(0.0, min(1.0, similarity)) # Assuming normalized embeddings >= 0

                if similarity >= similarity_threshold and faiss_id in speaker_map:
                    identified_name = speaker_map[faiss_id]
                    speaker_assignments[speaker] = identified_name
                    logger.info(f"  - Speaker {speaker}: Identified as '{identified_name}' (FAISS ID: {faiss_id}, Similarity: {similarity:.4f} >= {similarity_threshold:.4f})")
                else:
                    reason = ""
                    if similarity < similarity_threshold:
                        reason = f"Similarity {similarity:.4f} < {similarity_threshold:.4f}"
                    elif faiss_id not in speaker_map:
                        reason = f"FAISS ID {faiss_id} not found in speaker map (Similarity: {similarity:.4f})"
                    else:
                        reason = "Unknown reason" # Should not happen if conditions are exhaustive

                    logger.info(f"  - Speaker {speaker}: Marking as unknown. Reason: {reason}.")
                    unknown_speaker_count += 1
                    temp_id = f"UNKNOWN_{unknown_speaker_count}"
                    speaker_assignments[speaker] = temp_id
                    new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': query_embeddings_np[i], 'original_label': speaker})

        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            # Fallback: assign UNKNOWN to any remaining speakers in this batch
            processed_speakers = set(speaker_assignments.keys())
            for idx, speaker in enumerate(query_speakers_list):
                if speaker not in processed_speakers:
                    logger.warning(f"Assigning UNKNOWN to {speaker} due to FAISS search error.")
                    unknown_speaker_count += 1
                    temp_id = f"UNKNOWN_{unknown_speaker_count}_SEARCH_ERROR"
                    speaker_assignments[speaker] = temp_id
                    # Get the original embedding for this speaker
                    original_embedding = embeddings.get(speaker)
                    if isinstance(original_embedding, np.ndarray) and original_embedding.ndim == 1:
                        new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': original_embedding.astype(np.float32), 'original_label': speaker})

    logger.info("Speaker identification complete.")
    # Sort new speakers by their temporary ID number for consistent UI display
    try:
        new_speaker_embeddings_info.sort(key=lambda x: int(x['temp_id'].split('_')[-1]) if x['temp_id'].startswith("UNKNOWN_") and x['temp_id'].split('_')[-1].isdigit() else float('inf'))
    except (ValueError, IndexError, KeyError):
         logger.warning("Could not sort new speakers numerically by temp ID. Using original detection order.")

    return speaker_assignments, new_speaker_embeddings_info


# --- update_transcript_speakers (Keep as is) ---
def update_transcript_speakers(transcript_data: list, speaker_assignments: Dict[str, str]) -> list:
    """Updates word-level speaker labels in the transcript based on assignments."""
    logger.info("Updating transcript with identified speaker names or temp IDs...")
    updated_data = []
    if not isinstance(transcript_data, list):
        logger.error("Invalid transcript data format for updating speakers (expected list). Returning empty list.")
        return []

    processed_word_count = 0
    updated_word_count = 0

    for segment_idx, segment in enumerate(transcript_data):
        if not isinstance(segment, dict):
            logger.warning(f"Segment {segment_idx} is not a dictionary, keeping as is.")
            updated_data.append(segment)
            continue

        updated_segment = segment.copy()
        updated_words = []

        if 'words' in segment and isinstance(segment['words'], list):
            for word_idx, word_info in enumerate(segment['words']):
                processed_word_count += 1
                if not isinstance(word_info, dict):
                    logger.warning(f"Word {word_idx} in segment {segment_idx} is not a dictionary, keeping as is.")
                    updated_words.append(word_info)
                    continue

                updated_word_info = word_info.copy()
                original_speaker = word_info.get('speaker') # e.g., SPEAKER_01, UNKNOWN, etc.

                # Use the assignment map to get the final label (identified name or temp ID)
                # Default to original if not found in map (shouldn't happen if identify_speakers worked)
                new_speaker_label = speaker_assignments.get(original_speaker, original_speaker)
                update_made = (new_speaker_label != original_speaker)

                if original_speaker is None:
                    # Handle words that somehow missed getting a speaker label earlier
                    new_speaker_label = "SPEAKER_MISSING" # Assign a specific label
                    update_made = True
                    logger.warning(f"Word '{word_info.get('word', 'N/A')}' at index {word_idx} (segment {segment_idx}) had missing speaker label.")

                updated_word_info['speaker'] = new_speaker_label
                updated_words.append(updated_word_info)
                if update_made:
                    updated_word_count += 1

            updated_segment['words'] = updated_words
        else:
            # Ensure 'words' key exists even if segment had none originally
            updated_segment['words'] = []
            logger.debug(f"Segment starting at {segment.get('start', 'N/A')}s has no valid 'words' list.")


        updated_data.append(updated_segment)

    logger.info(f"Transcript update complete. Processed {processed_word_count} words, updated speaker label for {updated_word_count} words.")
    return updated_data


# --- run_speaker_identification (Modified to add timestamps and save snippets) ---
def run_speaker_identification(
    combined_transcript: list,
    embedding_model: Any,
    emb_dim: int,
    audio_path: str, # Full path to the original audio file
    output_dir: str, # Base output directory for the job (e.g., /app/transcripts_output/job_id)
    processing_device: str,
    min_segment_duration: float,
    similarity_threshold: float,
    faiss_index_path: str,
    speaker_map_path: str
    ) -> Tuple[Optional[list], list]: # Returns identified transcript and list of new speaker info WITH TIMESTAMPS
    """
    Runs speaker identification, returning the updated transcript and info needed for potential enrollment.
    The returned info includes embeddings and start/end times for unknown speakers.
    Also saves audio snippets for unknown speakers.

    Returns:
        Tuple containing:
            - The identified transcript (list or None if input was empty/invalid).
            - List of dictionaries for speakers needing enrollment (potentially empty).
              Each dict: {'temp_id': ..., 'embedding': ..., 'original_label': ..., 'start_time': ..., 'end_time': ...}
    """
    logger.info("--- Starting Speaker Identification Process ---")
    identified_transcript = combined_transcript # Default if ID fails
    new_speaker_info_for_enrollment = []
    faiss_index = None
    speaker_map = None

    if not embedding_model:
        logger.warning("Skipping Speaker Identification (embedding model not loaded).")
        return identified_transcript, []

    if not combined_transcript:
        logger.warning("Skipping Speaker Identification (no transcript data).")
        return identified_transcript, []

    start_id_time = time.time()
    waveform, sample_rate = None, None
    speaker_audio_segments = None
    speaker_embeddings = None

    try:
        # Load the full audio waveform once
        waveform, sample_rate = audio_processing.load_audio(audio_path)
        if waveform is None: raise ValueError("Audio loading failed.")

        # Load or create speaker database (index and map)
        faiss_index = persistence.load_or_create_faiss_index(faiss_index_path, emb_dim)
        speaker_map = persistence.load_or_create_speaker_map(speaker_map_path)

        # Extract audio segments for embedding generation
        speaker_audio_segments, effective_sr = audio_processing.extract_speaker_audio_segments(
            combined_transcript, waveform, sample_rate, min_segment_duration
        )
        # Ensure we use the effective sample rate if resampling occurred
        if effective_sr != sample_rate:
            logger.info(f"Using effective sample rate {effective_sr}Hz for subsequent processing.")
            sample_rate = effective_sr


        if not speaker_audio_segments:
            logger.warning("No sufficient audio segments extracted. Skipping embedding generation and identification.")
            identified_transcript = combined_transcript
            new_speaker_info_for_enrollment = []
        else:
            # Generate embeddings for extracted segments
            speaker_embeddings = get_speaker_embeddings(
                speaker_audio_segments, embedding_model, processing_device
            )

            if speaker_embeddings:
                # Identify speakers against the database
                speaker_assignments, new_speaker_embeddings_info = identify_speakers(
                    speaker_embeddings, faiss_index, speaker_map, similarity_threshold
                )
                # Update the transcript with identified/temporary labels
                identified_transcript = update_transcript_speakers(
                    combined_transcript, speaker_assignments
                )

                # --- START: Add Timestamps and Save Snippets for Unknown Speakers ---
                if new_speaker_embeddings_info:
                    logger.info(f"Found {len(new_speaker_embeddings_info)} potential new speakers. Determining timestamps and saving snippets...")
                    snippet_dir = os.path.join(output_dir, SNIPPETS_SUBDIR)
                    os.makedirs(snippet_dir, exist_ok=True)

                    # Keep track of saved snippets to avoid duplicates if a speaker has multiple segments
                    saved_snippets_for_temp_id = set()
                    temp_new_speaker_info_with_times = [] # Build the final list here

                    for speaker_data in new_speaker_embeddings_info:
                        temp_id = speaker_data['temp_id']
                        original_label = speaker_data['original_label']
                        embedding = speaker_data['embedding'] # Keep the embedding

                        # Find min start and max end time for this speaker's original label
                        min_start_time = float('inf')
                        max_end_time = float('-inf')
                        found_words = False
                        for segment in combined_transcript: # Iterate original transcript for timings
                            if 'words' in segment:
                                for word in segment['words']:
                                    if word.get('speaker') == original_label and 'start' in word and 'end' in word:
                                        min_start_time = min(min_start_time, word['start'])
                                        max_end_time = max(max_end_time, word['end'])
                                        found_words = True

                        if not found_words:
                            logger.warning(f"Could not find any words with timings for original label '{original_label}' (temp_id: {temp_id}). Cannot determine timestamp range.")
                            # Optionally add with default times or skip
                            # continue # Skip this speaker if no times found
                            min_start_time = 0.0 # Default start
                            max_end_time = min(SNIPPET_MAX_DURATION_SEC, 5.0) # Default end (e.g., first 5 seconds)
                            logger.warning(f"Using default time range {min_start_time}-{max_end_time} for {temp_id}")


                        # Add the info with timestamps to the list to be returned
                        temp_new_speaker_info_with_times.append({
                            'temp_id': temp_id,
                            'embedding': embedding, # Include embedding for saving enrollment info later
                            'original_label': original_label,
                            'start_time': min_start_time if min_start_time != float('inf') else 0.0, # Use calculated or default
                            'end_time': max_end_time if max_end_time != float('-inf') else min_start_time + SNIPPET_MAX_DURATION_SEC # Use calculated or default duration
                        })

                        # Save snippet (if not already saved for this temp_id)
                        if temp_id not in saved_snippets_for_temp_id:
                             # Use the determined min_start_time for snippet saving logic
                             snippet_start_time = min_start_time if min_start_time != float('inf') else 0.0
                             logger.info(f"Saving snippet for {temp_id} (orig: {original_label}) from time {snippet_start_time:.2f}s")
                             snippet_filename_base = f"snippet_{temp_id}" # Use temp_id in filename
                             saved_path = audio_processing.save_speaker_snippet(
                                 full_waveform=waveform,
                                 sample_rate=sample_rate,
                                 start_sec=snippet_start_time,
                                 # Use a fixed max duration for the snippet, don't rely on max_end_time here
                                 end_sec=snippet_start_time + SNIPPET_MAX_DURATION_SEC,
                                 max_snippet_sec=SNIPPET_MAX_DURATION_SEC, # Explicitly pass max duration
                                 output_dir=snippet_dir,
                                 filename_base=snippet_filename_base
                             )
                             if saved_path:
                                 saved_snippets_for_temp_id.add(temp_id)
                             else:
                                 logger.warning(f"Failed to save snippet for {temp_id}")

                    # Replace the original list with the one containing timestamps
                    new_speaker_info_for_enrollment = temp_new_speaker_info_with_times
                    # --- END: Add Timestamps and Save Snippets ---

            else:
                logger.warning("No speaker embeddings generated. Skipping identification and snippet saving.")
                identified_transcript = combined_transcript # Keep combined version
                new_speaker_info_for_enrollment = [] # Ensure it's empty

        # --- Enrollment is NOT done here ---
        # The calling function (run_full_pipeline) will handle enrollment based on stage

        end_id_time = time.time()
        logger.info(f"Speaker Identification process finished. Found {len(new_speaker_info_for_enrollment)} potential new speakers. Took {end_id_time - start_id_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during speaker identification steps: {e}", exc_info=True)
        # Fallback to original combined transcript if ID fails
        identified_transcript = combined_transcript
        new_speaker_info_for_enrollment = [] # Clear enrollment info on error
        # Re-raise the exception to be caught by the main pipeline function
        raise RuntimeError(f"Speaker Identification failed: {e}") from e

    finally:
        # Cleanup local variables for this function
        # Waveform is needed for snippets, cleanup handled by caller (run_full_pipeline)
        # del waveform
        if speaker_audio_segments: del speaker_audio_segments
        if speaker_embeddings: del speaker_embeddings
        # FAISS index/map are handled by the caller or persistence

    # Return the transcript with identified/temp labels and the list of potential new speakers (now with timestamps)
    return identified_transcript, new_speaker_info_for_enrollment


# --- enroll_new_speakers_cli (Keep as is) ---
def enroll_new_speakers_cli(
    new_speaker_info: List[Dict[str, Any]],
    faiss_index: Any,
    speaker_map: Dict[int, str],
    faiss_index_path: str,
    speaker_map_path: str,
    ) -> bool:
    """
    Handles interactive enrollment of new speakers via CLI and saves updates.
    """
    if not new_speaker_info:
        logger.info("No new speakers detected requiring enrollment.")
        return False # No changes made

    if faiss_index is None or speaker_map is None:
        logger.error("FAISS index or speaker map missing. Cannot proceed with enrollment.")
        return False

    # --- Interactive Mode ---
    logger.info("\n--- Speaker Enrollment (CLI) ---")
    print("\nThe following speakers were not identified in the database.")
    print("Please provide a name to enroll them, or leave blank to skip.")

    changes_made = False
    speakers_to_enroll = [] # Store tuples of (name, embedding, temp_id)

    for speaker_data in new_speaker_info:
        temp_id = speaker_data.get('temp_id', 'UNKNOWN_ID')
        embedding = speaker_data.get('embedding')
        original_label = speaker_data.get('original_label', 'N/A')

        # Validate embedding before prompting
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            logger.warning(f"Skipping enrollment prompt for {temp_id} (orig: {original_label}) due to invalid embedding (Dim: {embedding.ndim if isinstance(embedding, np.ndarray) else 'N/A'}).")
            continue

        # Ensure embedding is float32 for FAISS
        embedding = embedding.astype(np.float32)

        prompt = f"Enroll speaker {temp_id} (originally '{original_label}')? Enter name (or press Enter to skip): "
        try:
            # This input() call only happens if interactive is True
            entered_name = input(prompt).strip()
            if entered_name:
                logger.info(f"User provided name '{entered_name}' for {temp_id}.")
                speakers_to_enroll.append((entered_name, embedding, temp_id))
                # Mark potential change, actual change happens on save
                # changes_made = True # We set this to True only if saving succeeds
            else:
                logger.info(f"Skipping enrollment for {temp_id}.")
        except EOFError:
            logger.warning("\nEOF detected during input. Skipping remaining enrollment.")
            print("\nSkipping remaining enrollment.")
            break # Stop prompting if input stream closes
        except Exception as e:
             logger.error(f"Error during input for {temp_id}: {e}. Skipping this speaker.", exc_info=True)
             print(f"An error occurred processing input for {temp_id}. Skipping.")

    # --- Process enrollments if any names were provided ---
    if speakers_to_enroll:
        logger.info(f"Attempting to enroll {len(speakers_to_enroll)} new speakers into the database.")
        num_enrolled = 0
        try:
            # Add embeddings to FAISS in batch
            embeddings_np = np.array([emb for _, emb, _ in speakers_to_enroll])
            start_index = faiss_index.ntotal
            faiss_index.add(embeddings_np)
            end_index = faiss_index.ntotal
            logger.info(f"Added {embeddings_np.shape[0]} embeddings to FAISS. Index size: {start_index} -> {end_index}")

            if end_index != start_index + len(speakers_to_enroll):
                logger.error(f"FAISS index size mismatch after adding embeddings! Expected {start_index + len(speakers_to_enroll)}, got {end_index}. Enrollment may be corrupted. Aborting save.")
                return False # Indicate failure, no changes saved

            # Update speaker map
            for i, (name, _, temp_id) in enumerate(speakers_to_enroll):
                new_faiss_id = start_index + i
                if new_faiss_id >= end_index: # Safety check against mismatch
                    logger.error(f"Calculated FAISS ID {new_faiss_id} is out of bounds for speaker '{name}' ({temp_id}). Skipping map update for this speaker.")
                    continue

                if new_faiss_id in speaker_map:
                    logger.warning(f"Overwriting existing map entry for FAISS ID {new_faiss_id} ('{speaker_map[new_faiss_id]}') with new speaker '{name}' ({temp_id}).")
                speaker_map[new_faiss_id] = name
                logger.info(f"   -> Mapped FAISS ID {new_faiss_id} to '{name}'.")
                num_enrolled += 1

            if num_enrolled != len(speakers_to_enroll):
                logger.warning(f"Only {num_enrolled} out of {len(speakers_to_enroll)} speakers were successfully mapped due to potential errors.")

            # Save the updated index and map *only if* mapping was at least partially successful
            if num_enrolled > 0:
                logger.info("Saving updated speaker database (FAISS index and map)...")
                # Use persistence functions directly if imported, or module.function if not
                persistence.save_faiss_index(faiss_index, faiss_index_path)
                persistence.save_speaker_map(speaker_map, speaker_map_path)

                # --- Optional Sanity Check ---
                final_index_size = faiss_index.ntotal
                final_map_size = len(speaker_map)
                if final_index_size != final_map_size:
                    logger.critical(f"CRITICAL WARNING: Mismatch after save! FAISS index size ({final_index_size}) != Speaker map size ({final_map_size}). Manual check required: {faiss_index_path}, {speaker_map_path}")
                    print(f"\nCRITICAL WARNING: Speaker database potentially inconsistent after saving. Check logs and files.")
                else:
                    logger.info(f"Enrollment successful. Database updated with {num_enrolled} speakers. Index/Map size: {final_index_size}.")

                changes_made = True # Indicate that the database files were actually modified
            else:
                logger.warning("No speakers were successfully mapped, skipping database save.")
                changes_made = False

            return changes_made

        except Exception as e:
            logger.error(f"Error during bulk enrollment or saving: {e}", exc_info=True)
            print("\nAn error occurred during enrollment. Database might not be updated correctly. Check logs.")
            return False # Indicate failure or partial success, no changes saved in this case

    else: # No speakers were entered by the user
        logger.info("No speakers were selected for enrollment by the user.")
        return False # No changes made


# --- enroll_speakers_programmatic (Keep as is) ---
def enroll_speakers_programmatic(
    enrollment_map: Dict[str, str], # Map of temp_id -> user_provided_name
    new_speaker_info: List[Dict[str, Any]], # List generated by identify_speakers (MUST contain embeddings)
    faiss_index: Any,
    speaker_map: Dict[int, str],
    faiss_index_path: str,
    speaker_map_path: str
    ) -> bool:
    """
    Enrolls new speakers programmatically based on a mapping of temp IDs to names.
    Updates and saves the FAISS index and speaker map.
    """
    logger.info("--- Starting Programmatic Speaker Enrollment ---")
    if not enrollment_map:
        logger.info("No speakers provided in the enrollment map. Skipping enrollment.")
        return False

    if faiss_index is None or speaker_map is None:
        logger.error("FAISS index or speaker map missing. Cannot proceed with programmatic enrollment.")
        return False

    speakers_to_enroll = [] # List of (name, embedding) tuples
    temp_id_to_name = enrollment_map

    # Find the corresponding embedding for each speaker in the enrollment map
    for speaker_data in new_speaker_info:
        temp_id = speaker_data.get('temp_id')
        embedding = speaker_data.get('embedding')

        # Check if this temp_id is one the user wants to enroll
        if temp_id in temp_id_to_name:
            user_provided_name = temp_id_to_name[temp_id]
            # Validate the embedding
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                logger.info(f"Preparing to enroll '{user_provided_name}' (from {temp_id}).")
                speakers_to_enroll.append((user_provided_name, embedding.astype(np.float32)))
            else:
                logger.warning(f"Skipping enrollment for '{user_provided_name}' ({temp_id}) due to invalid embedding.")

    # --- Process enrollments ---
    if speakers_to_enroll:
        logger.info(f"Attempting to enroll {len(speakers_to_enroll)} new speakers programmatically.")
        num_enrolled = 0
        try:
            # Add embeddings to FAISS in batch
            embeddings_np = np.array([emb for _, emb in speakers_to_enroll])
            start_index = faiss_index.ntotal
            faiss_index.add(embeddings_np)
            end_index = faiss_index.ntotal
            logger.info(f"Added {embeddings_np.shape[0]} embeddings to FAISS. Index size: {start_index} -> {end_index}")

            if end_index != start_index + len(speakers_to_enroll):
                logger.error(f"FAISS index size mismatch after adding embeddings! Expected {start_index + len(speakers_to_enroll)}, got {end_index}. Enrollment may be corrupted. Aborting save.")
                return False

            # Update speaker map
            for i, (name, _) in enumerate(speakers_to_enroll):
                new_faiss_id = start_index + i
                if new_faiss_id >= end_index: # Safety check
                    logger.error(f"Calculated FAISS ID {new_faiss_id} is out of bounds for speaker '{name}'. Skipping map update.")
                    continue

                if new_faiss_id in speaker_map:
                    logger.warning(f"Overwriting existing map entry for FAISS ID {new_faiss_id} ('{speaker_map[new_faiss_id]}') with new speaker '{name}'.")
                speaker_map[new_faiss_id] = name
                logger.info(f"   -> Mapped FAISS ID {new_faiss_id} to '{name}'.")
                num_enrolled += 1

            if num_enrolled != len(speakers_to_enroll):
                logger.warning(f"Only {num_enrolled} out of {len(speakers_to_enroll)} speakers were successfully mapped.")

            # Save the updated index and map if enrollment happened
            if num_enrolled > 0:
                logger.info("Saving updated speaker database (FAISS index and map)...")
                # Use persistence functions directly if imported, or module.function if not
                persistence.save_faiss_index(faiss_index, faiss_index_path)
                persistence.save_speaker_map(speaker_map, speaker_map_path)
                # Optional Sanity Check
                final_index_size = faiss_index.ntotal
                final_map_size = len(speaker_map)
                if final_index_size != final_map_size:
                    logger.critical(f"CRITICAL WARNING: Mismatch after save! FAISS index size ({final_index_size}) != Speaker map size ({final_map_size}). Manual check required.")
                else:
                    logger.info(f"Programmatic enrollment successful. Database updated with {num_enrolled} speakers. Index/Map size: {final_index_size}.")
                return True # Database updated
            else:
                logger.warning("No speakers were successfully mapped programmatically, skipping database save.")
                return False # Database not updated

        except Exception as e:
            logger.error(f"Error during programmatic enrollment or saving: {e}", exc_info=True)
            return False # Indicate failure

    else:
        logger.info("No valid speakers found to enroll based on the provided map.")
        return False # No changes made

