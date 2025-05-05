# text_processing.py
import re
import logging
from typing import List, Dict, Any, Tuple, Optional # <-- Import Tuple and Optional here

# Import external libraries (ensure they are installed)
try:
    from cucco import Cucco
except ImportError:
    Cucco = None # Define as None if not installed
try:
    import inflect
except ImportError:
    inflect = None # Define as None if not installed

import traceback
from collections import defaultdict


logger = logging.getLogger(__name__)

# Initialize globally or pass instances if preferred
# Passing instances is generally better for testability
# cucco_instance = Cucco() if Cucco else None
# inflect_engine = inflect.engine() if inflect else None

def format_punctuated_output(results: List[Dict[str, Any]]) -> str:
    """
    Reconstructs text from the punctuation pipeline output (token classification).
    Handles subwords and applies punctuation/capitalization based on entity labels.
    Assumes HuggingFace pipeline with aggregation_strategy="simple".
    Example entity labels: 'PERIOD', 'COMMA', 'QUESTION', 'LABEL_X' (adjust as needed).
    """
    text = ''
    last_word_end = -1 # Track end position to handle spacing correctly

    logger.debug(f"Formatting {len(results)} punctuation results.")
    if not results: return ""

    # Define punctuation marks that should attach to the preceding word
    ATTACHING_PUNCTUATION_LABELS = {
        'PERIOD', 'PUNC_PERIOD',
        'QUESTION', 'PUNC_QUESTION',
        'COMMA', 'PUNC_COMMA',
        'EXCLAMATION', 'PUNC_EXCLAMATION',
        'COLON', 'PUNC_COLON',
        'SEMICOLON', 'PUNC_SEMICOLON',
    }
    # Define labels to ignore (no specific punctuation action needed)
    IGNORE_LABELS = {'O', 'LABEL_0', 'LABEL_1'} # <-- Added LABEL_1 here

    try:
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid item in punctuation results: {item}")
                continue

            word = item.get('word', '').strip()
            # Prefer entity_group, fallback to entity. Standardize case. Handle None.
            entity = str(item.get('entity_group', item.get('entity', 'O'))).upper()
            score = item.get('score', 0.0)
            start = item.get('start') # Start/end may not be reliable
            end = item.get('end')

            if not word: continue # Skip empty words

            # --- Handle Subwords (Heuristic based on '##' or contiguity) ---
            is_subword = word.startswith('##') or (start is not None and last_word_end is not None and start == last_word_end)
            if word.startswith('##'):
                word = word[2:]

            # --- Spacing ---
            # Add space if not the first word AND not a subword immediately following the previous one.
            if text and not is_subword and not text.endswith(('-', "'")):
                # Also avoid space *before* punctuation marks that should attach to the word
                 if entity not in ATTACHING_PUNCTUATION_LABELS:
                      text += ' '

            # --- Capitalization (Heuristic) ---
            # Capitalize if it's the first word OR follows sentence-ending punctuation.
            if i == 0 or (text and re.search(r'[.?!]\s*$', text)):
                 if word: word = word.capitalize()

            # Append the word
            text += word

            # --- Apply Punctuation based on Label ---
            # Map model-specific labels to actual punctuation marks.
            punct_map = {
                'PERIOD': '.', 'PUNC_PERIOD': '.',
                'QUESTION': '?', 'PUNC_QUESTION': '?',
                'COMMA': ',', 'PUNC_COMMA': ',',
                'EXCLAMATION': '!', 'PUNC_EXCLAMATION': '!',
                'COLON': ':', 'PUNC_COLON': ':',
                'SEMICOLON': ';', 'PUNC_SEMICOLON': ';',
                # Add other mappings as needed (e.g., HYPHEN, QUOTE)
            }

            if entity in punct_map:
                text += punct_map[entity]
            # Check if the label should be ignored or logged as unhandled
            elif entity not in IGNORE_LABELS:
                logger.debug(f"Unhandled punctuation entity label '{entity}' for word '{word}' (Score: {score:.2f}).")


            # Update last word end position
            if end is not None:
                 last_word_end = end

    except Exception as e:
        logger.error(f"Error during punctuation formatting: {e}", exc_info=True)
        # Fallback: simple join, which will likely be wrong without punctuation/capitalization
        return " ".join([item.get('word', '') for item in results if isinstance(item, dict)])

    # --- Final Cleanup ---
    # Remove potential spaces before punctuation added above
    text = re.sub(r'\s+([.,?!:;])', r'\1', text)
    # Ensure single spaces between words
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # logger.debug(f"Formatted text sample: '{text[:100]}...'") # Reduce log verbosity
    return text


# Corrected type hint import
def apply_punctuation(transcript_data: list, punctuation_pipeline: Any, chunk_size: int = 256) -> List[Tuple[str, str]]:
    """Applies punctuation restoration model to speaker turns in the transcript."""
    logger.info("\n--- Applying Punctuation ---")

    if not punctuation_pipeline:
        logger.error("Punctuation pipeline not available. Skipping punctuation.")
        # Return data in a similar format but without punctuation?
        # Create turns with raw text:
        punctuated_turns = []
        current_speaker = None
        current_words = []
        if transcript_data: # Check if there's any data
            for segment in transcript_data:
                if 'words' not in segment or not isinstance(segment['words'], list): continue
                for word_info in segment['words']:
                    if not isinstance(word_info, dict): continue
                    speaker = word_info.get('speaker', 'UNKNOWN')
                    word = word_info.get('word', '').strip()
                    if not word: continue

                    if speaker != current_speaker and current_words:
                        raw_text = " ".join(current_words)
                        punctuated_turns.append((current_speaker, raw_text))
                        current_words = []
                    current_speaker = speaker
                    current_words.append(word)

            # Process the last turn
            if current_words:
                raw_text = " ".join(current_words)
                punctuated_turns.append((current_speaker, raw_text))
        logger.warning("Punctuation pipeline missing. Returning turns with raw, unpunctuated text.")
        return punctuated_turns


    punctuated_turns = [] # List of (speaker, text) tuples
    current_speaker = None
    current_words = []
    total_words_processed = 0
    turn_count = 0

    if not transcript_data:
         logger.warning("Input transcript data is empty. No punctuation to apply.")
         return []

    # --- Iterate through words and group by speaker ---
    for segment in transcript_data:
        if 'words' not in segment or not isinstance(segment['words'], list): continue
        for word_info in segment['words']:
            if not isinstance(word_info, dict): continue

            speaker = word_info.get('speaker', 'UNKNOWN') # Default speaker if missing
            word = word_info.get('word', '').strip()
            if not word: continue # Skip empty word entries

            # --- Process Previous Turn When Speaker Changes ---
            if speaker != current_speaker and current_words:
                turn_count += 1
                logger.info(f"Processing turn {turn_count} for {current_speaker} ({len(current_words)} words)")
                full_turn_text = ""
                try:
                    # Process the turn in chunks
                    for i in range(0, len(current_words), chunk_size):
                        chunk_words = current_words[i:i+chunk_size]
                        raw_text_chunk = " ".join(chunk_words)
                        if not raw_text_chunk: continue # Skip empty chunks

                        logger.debug(f"  Punctuating chunk ({len(chunk_words)} words): '{raw_text_chunk[:80]}...'")
                        # Process chunk using the pipeline
                        try:
                             processed_results = punctuation_pipeline(raw_text_chunk)
                             # Format the results of the chunk
                             punctuated_chunk = format_punctuated_output(processed_results)
                             full_turn_text += punctuated_chunk + " " # Add space between chunks
                             total_words_processed += len(chunk_words)
                        except Exception as pipe_e:
                             logger.error(f"    Error during punctuation pipeline call for chunk: {pipe_e}", exc_info=True)
                             logger.warning(f"    Appending raw chunk text for {current_speaker} due to error.")
                             full_turn_text += raw_text_chunk + " " # Fallback to raw text for the chunk

                    punctuated_turns.append((current_speaker, full_turn_text.strip()))
                    # logger.debug(f"  Finished turn for {current_speaker}. Result: '{full_turn_text.strip()[:80]}...'") # Reduce log verbosity

                except Exception as e:
                    logger.error(f"  Unexpected error processing turn for {current_speaker}: {e}. Turn may be incomplete or use raw text.", exc_info=True)
                    # Append raw text as fallback for the entire turn if chunking loop fails badly
                    if not full_turn_text: # Only if nothing was processed
                         punctuated_turns.append((current_speaker, " ".join(current_words)))

                current_words = [] # Reset for the new speaker

            # --- Update Current Speaker and Add Word ---
            current_speaker = speaker
            current_words.append(word)

    # --- Process the Final Turn ---
    if current_words:
        turn_count += 1
        logger.info(f"Processing final turn {turn_count} for {current_speaker} ({len(current_words)} words)")
        full_turn_text = ""
        try:
            for i in range(0, len(current_words), chunk_size):
                chunk_words = current_words[i:i+chunk_size]
                raw_text_chunk = " ".join(chunk_words)
                if not raw_text_chunk: continue

                logger.debug(f"  Punctuating final chunk ({len(chunk_words)} words): '{raw_text_chunk[:80]}...'")
                try:
                    processed_results = punctuation_pipeline(raw_text_chunk)
                    punctuated_chunk = format_punctuated_output(processed_results)
                    full_turn_text += punctuated_chunk + " "
                    total_words_processed += len(chunk_words)
                except Exception as pipe_e:
                     logger.error(f"    Error during punctuation pipeline call for final chunk: {pipe_e}", exc_info=True)
                     logger.warning(f"    Appending raw chunk text for final turn {current_speaker} due to error.")
                     full_turn_text += raw_text_chunk + " " # Fallback

            punctuated_turns.append((current_speaker, full_turn_text.strip()))
            # logger.debug(f"  Finished final turn for {current_speaker}. Result: '{full_turn_text.strip()[:80]}...'") # Reduce log verbosity

        except Exception as e:
            logger.error(f"  Unexpected error processing final turn for {current_speaker}: {e}.", exc_info=True)
            if not full_turn_text:
                 punctuated_turns.append((current_speaker, " ".join(current_words))) # Fallback

    logger.info(f"Punctuation application complete. Processed approx {total_words_processed} words into {len(punctuated_turns)} turns.")
    return punctuated_turns


# Corrected type hint import
def normalize_text_custom(
    text: str,
    cucco_instance: Optional[Cucco], # Expect initialized instances or None
    inflect_engine: Optional[inflect.engine],
    normalize_numbers: bool = False,
    remove_fillers: bool = False,
    filler_words: Optional[list] = None
) -> str:
    """
    Applies selected text normalization rules using provided instances.
    """
    # Use a sub-logger if desired, or keep using the main text_processing logger
    # logger = logging.getLogger(__name__ + ".normalize")
    if filler_words is None: filler_words = []

    # Ensure necessary libraries are available if options are enabled
    if normalize_numbers and not inflect_engine:
        logger.warning("Inflect engine not available, skipping number normalization.")
        normalize_numbers = False # Disable if engine missing
    if remove_fillers and not cucco_instance: # Cucco often used for whitespace cleanup after removal
         logger.warning("Cucco instance not available, filler removal might leave extra spaces.")
         # Proceed anyway, but final cleanup is important

    if not isinstance(text, str):
        logger.error(f"Invalid input type for normalization: expected str, got {type(text)}. Returning input as is.")
        return text

    normalized_text = text
    # logger.debug(f"Starting normalization for text: '{text[:100]}...' (Num: {normalize_numbers}, Fillers: {remove_fillers})") # Reduce log verbosity

    # --- 1. Cucco Basic Cleanup (Whitespace) ---
    # Run this early if needed, or at the end
    if cucco_instance:
        try:
            # Only apply whitespace normalization for now
            normalized_text = cucco_instance.normalize(normalized_text, ['remove_extra_white_spaces'])
            # logger.debug("Applied Cucco 'remove_extra_white_spaces'.")
        except Exception as e:
            logger.warning(f"Cucco normalization ('remove_extra_white_spaces') failed: {e}. Continuing.", exc_info=False)


    # --- 2. Number to Words Conversion (using inflect) ---
    if normalize_numbers and inflect_engine:
        try:
            def replace_num(match):
                num_str = match.group(0)
                try:
                    num_str_no_comma = num_str.replace(',', '')
                    # Check if inflect can handle it (basic integer/float check)
                    if re.fullmatch(r'-?\d+(\.\d+)?', num_str_no_comma):
                        num_word = inflect_engine.number_to_words(num_str_no_comma)
                        # Handle potential "and" insertion by inflect for decimals/large numbers
                        num_word = num_word.replace(" and ", " ") # Optional: remove 'and' if not desired
                        return num_word
                    else:
                        return num_str # Return original if not a simple number
                except Exception:
                     # logger.warning(f"Inflect failed to convert number '{num_str}': {e}. Keeping original.", exc_info=False) # Reduce noise
                     return num_str # Keep original on conversion error

            # Regex to find sequences of digits, possibly with commas or a decimal point, as whole words.
            # Handles integers, comma-separated integers, decimals.
            number_pattern = r'\b-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b-?\d+\b'
            original_text_numbers = normalized_text
            normalized_text = re.sub(number_pattern, replace_num, normalized_text)
            # if normalized_text != original_text_numbers:
            #     logger.debug("Number to words conversion applied.")
        except Exception as e:
             logger.error(f"Error during number to words conversion step (re.sub): {e}.", exc_info=True)


    # --- 3. Filler Word Removal (case-insensitive) ---
    if remove_fillers and filler_words:
        try:
            # logger.debug(f"Attempting to remove {len(filler_words)} filler words (case-insensitive)...")
            # Sort by length descending to match longer phrases first (e.g., "you know" before "know")
            filler_words_sorted = sorted(filler_words, key=len, reverse=True)
            # Escape regex special characters in filler words
            escaped_fillers = [re.escape(word) for word in filler_words_sorted]

            # Build pattern: word boundary, any filler, word boundary (case-insensitive)
            # This simpler pattern removes the word and relies on later cleanup for spacing.
            pattern_str = r'\b(?:' + '|'.join(escaped_fillers) + r')\b'

            original_text_fillers = normalized_text
            # Perform case-insensitive replacement
            normalized_text = re.sub(pattern_str, '', normalized_text, flags=re.IGNORECASE)

            # Cleanup spacing after removal (crucial)
            if normalized_text != original_text_fillers:
                # logger.debug("Filler word removal regex applied. Cleaning up spacing...")
                # Consolidate multiple spaces to single spaces and strip ends
                normalized_text = ' '.join(normalized_text.split())
                # Fix spaces before punctuation (e.g., "word , word" -> "word, word")
                normalized_text = re.sub(r'\s+([.,?!:;])', r'\1', normalized_text)
                # logger.debug("Cleaned up spacing after filler removal.")
            # else:
                # logger.debug("No filler words found or removed.")
        except re.error as regex_e:
             logger.error(f"Regex error during filler word removal setup: {regex_e}. Skipping filler removal.", exc_info=True)
        except Exception as e:
             logger.error(f"Error during filler word removal: {e}. Skipping filler removal.", exc_info=True)

    elif remove_fillers and not filler_words:
        logger.warning("Filler word removal requested, but the filler_words list is empty.")


    # --- 4. Final Whitespace Cleanup (Run again after potential modifications) ---
    try:
        final_cleaned_text = ' '.join(normalized_text.split())
        normalized_text = final_cleaned_text
    except Exception as e:
         logger.error(f"Error during final whitespace cleanup: {e}", exc_info=True)

    # logger.debug(f"Normalization complete. Result: '{normalized_text[:100]}...'") # Reduce log verbosity
    return normalized_text
