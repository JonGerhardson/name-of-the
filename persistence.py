# persistence.py
import os
import json
import faiss
import numpy as np
import logging
import math # For SRT timestamps
# START <<< MODIFICATION >>> Import Optional
from typing import Optional, List, Dict, Any # Added Optional here
# END <<< MODIFICATION >>>

logger = logging.getLogger(__name__)

# --- FAISS / Speaker Map ---
# ... (load/save faiss index and speaker map functions remain the same) ...
def load_or_create_faiss_index(index_path, dimension):
    """Loads a FAISS index or creates a new one."""
    if os.path.exists(index_path):
        logger.info(f"Attempting to load existing FAISS index from {index_path}")
        try:
            index = faiss.read_index(index_path)
            if index.d != dimension:
                logger.warning(f"Index dimension ({index.d}) differs from model dimension ({dimension}). Creating new index.")
                index = faiss.IndexFlatIP(dimension) # Use Inner Product
            else:
                logger.info(f"FAISS index loaded successfully with {index.ntotal} embeddings (Dim: {index.d}).")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}. Creating a new one.", exc_info=True)
            index = faiss.IndexFlatIP(dimension)
    else:
        logger.info(f"FAISS index not found at {index_path}. Creating a new one with dimension {dimension}.")
        index = faiss.IndexFlatIP(dimension)
    return index

def save_faiss_index(index, index_path):
    """Saves the FAISS index to disk."""
    logger.info(f"Saving FAISS index to {index_path} with {index.ntotal} embeddings...")
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        logger.info("FAISS index saved successfully.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}", exc_info=True)

def load_or_create_speaker_map(map_path):
    """Loads the speaker name map (FAISS ID -> Name) from JSON."""
    if os.path.exists(map_path):
        logger.info(f"Loading speaker map from {map_path}")
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                speaker_map = json.load(f)
            # Convert keys back to integers
            speaker_map = {int(k): v for k, v in speaker_map.items()}
            logger.info(f"Speaker map loaded with {len(speaker_map)} entries.")
        except Exception as e:
            logger.error(f"Error loading speaker map: {e}. Creating a new one.", exc_info=True)
            speaker_map = {}
    else:
        logger.info(f"Speaker map not found at {map_path}. Creating a new one.")
        speaker_map = {}
    return speaker_map

def save_speaker_map(speaker_map, map_path):
    """Saves the speaker name map to JSON."""
    logger.info(f"Saving speaker map to {map_path} with {len(speaker_map)} entries...")
    try:
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        # Ensure keys are strings for JSON compatibility
        save_map = {str(k): v for k, v in speaker_map.items()}
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(save_map, f, indent=2)
        logger.info("Speaker map saved successfully.")
    except Exception as e:
        logger.error(f"Error saving speaker map: {e}", exc_info=True)


# --- Transcript Saving / Loading ---

def save_word_level_transcript(transcript_data, output_path):
    """Saves word-level transcript data (intermediate or final) to a JSON file."""
    logger.info(f"Saving word-level transcript JSON to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        logger.info("Word-level transcript JSON saved successfully.")
    except Exception as e:
        logger.error(f"Error saving word-level transcript JSON: {e}", exc_info=True)
        raise

def load_word_level_transcript(input_path):
    """Loads the word-level transcript data from a JSON file."""
    logger.info(f"Loading word-level transcript from {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Word-level transcript file not found: {input_path}")
        raise FileNotFoundError(f"Word-level transcript file not found: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        logger.info("Word-level transcript loaded successfully.")
        return transcript_data
    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from word-level transcript file {input_path}: {jde}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading word-level transcript JSON from {input_path}: {e}", exc_info=True)
        raise

def save_final_transcript(punctuated_turns, output_path):
    """Saves the final, formatted transcript (speaker: text block) to a text file."""
    logger.info(f"Saving final standard transcript (speaker: text block) to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if punctuated_turns:
                for speaker, text in punctuated_turns:
                    speaker_str = str(speaker) if speaker is not None else "UNKNOWN_SPEAKER"
                    text_str = str(text).strip() if text is not None else ""
                    f.write(f"[{speaker_str}]:\n{text_str}\n\n")
                logger.info(f"Final standard transcript saved successfully ({len(punctuated_turns)} speaker turns).")
            else:
                logger.warning("Final transcript data is empty or None. Saving an empty file.")
                f.write("")
    except Exception as e:
        logger.error(f"Error saving final standard transcript text file: {e}", exc_info=True)
        raise

def _format_timestamp(seconds: Optional[float], format_srt: bool = False) -> str:
    """Helper to format seconds into HH:MM:SS,ms or HH:MM:SS.ms format."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00,000" if format_srt else "00:00:00.000"
    seconds = float(seconds)
    milliseconds = math.floor((seconds - math.floor(seconds)) * 1000)
    total_seconds = math.floor(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    separator = "," if format_srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"

def save_transcript_with_timestamps(transcript_data, output_path):
    """Saves transcript with speaker labels and timestamps per turn."""
    logger.info(f"Saving transcript with timestamps to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            current_speaker = None
            current_turn_words = []
            turn_start_time = None
            turn_end_time = None

            for segment in transcript_data:
                if 'words' not in segment or not segment['words']:
                    continue
                for word_info in segment['words']:
                    speaker = word_info.get('speaker', 'UNKNOWN')
                    word = word_info.get('word', '').strip()
                    start = word_info.get('start')
                    end = word_info.get('end')

                    if not word: continue

                    if speaker != current_speaker:
                        if current_speaker is not None and current_turn_words and turn_start_time is not None and turn_end_time is not None:
                            start_fmt = _format_timestamp(turn_start_time)
                            end_fmt = _format_timestamp(turn_end_time)
                            f.write(f"[{current_speaker} ({start_fmt} - {end_fmt})]:\n")
                            f.write(" ".join(current_turn_words) + "\n\n")
                        current_speaker = speaker
                        current_turn_words = [word]
                        turn_start_time = start
                        turn_end_time = end
                    else:
                        current_turn_words.append(word)
                        if end is not None:
                            turn_end_time = end

            if current_speaker is not None and current_turn_words and turn_start_time is not None and turn_end_time is not None:
                start_fmt = _format_timestamp(turn_start_time)
                end_fmt = _format_timestamp(turn_end_time)
                f.write(f"[{current_speaker} ({start_fmt} - {end_fmt})]:\n")
                f.write(" ".join(current_turn_words) + "\n\n")

        logger.info("Transcript with timestamps saved successfully.")
    except Exception as e:
        logger.error(f"Error saving transcript with timestamps: {e}", exc_info=True)
        raise

def save_transcript_srt(transcript_data, output_path, words_per_line=10):
    """Saves transcript in SRT subtitle format."""
    logger.info(f"Saving transcript in SRT format to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            subtitle_index = 1
            all_words = []
            for segment in transcript_data:
                if 'words' in segment and segment['words']:
                    all_words.extend(segment['words'])

            if not all_words:
                 logger.warning("No words found in transcript data for SRT generation.")
                 return

            for i in range(0, len(all_words), words_per_line):
                chunk = all_words[i : i + words_per_line]
                start_time = chunk[0].get('start')
                end_time = chunk[-1].get('end')
                speaker = chunk[0].get('speaker', 'UNKNOWN')

                if start_time is None or end_time is None:
                    logger.warning(f"Skipping SRT entry starting around word '{chunk[0].get('word')}' due to missing timestamps.")
                    continue

                start_fmt = _format_timestamp(start_time, format_srt=True)
                end_fmt = _format_timestamp(end_time, format_srt=True)
                line_text = " ".join([w.get('word', '').strip() for w in chunk if w.get('word')])

                f.write(f"{subtitle_index}\n")
                f.write(f"{start_fmt} --> {end_fmt}\n")
                f.write(f"[{speaker}]: {line_text}\n\n")
                subtitle_index += 1

        logger.info(f"SRT transcript saved successfully ({subtitle_index - 1} entries).")
    except Exception as e:
        logger.error(f"Error saving SRT transcript: {e}", exc_info=True)
        raise

def save_segment_level_json(transcript_data, output_path):
    """
    Saves the transcript as a JSON file containing segment-level information
    (start, end, text, speaker), omitting the detailed word list.
    """
    logger.info(f"Saving segment-level transcript JSON to {output_path}...")
    segment_level_data = []
    try:
        for segment in transcript_data:
            if 'start' in segment and 'end' in segment and 'words' in segment:
                segment_text = " ".join([w.get('word', '').strip() for w in segment['words'] if w.get('word')])
                primary_speaker = segment['words'][0].get('speaker', 'UNKNOWN') if segment['words'] else 'UNKNOWN'
                segment_level_data.append({
                    "start": segment.get('start'),
                    "end": segment.get('end'),
                    "text": segment_text,
                    "speaker": primary_speaker
                })
            else:
                logger.warning(f"Skipping segment due to missing keys (start, end, or words): {segment}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segment_level_data, f, indent=2, ensure_ascii=False)
        logger.info("Segment-level transcript JSON saved successfully.")

    except Exception as e:
        logger.error(f"Error saving segment-level transcript JSON: {e}", exc_info=True)
        raise

