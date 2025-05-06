# audio_processing.py
import os
import logging
import torch
import torchaudio
import soundfile as sf # For saving audio snippets
from typing import Dict, Tuple, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# --- Existing load_audio function (ensure it returns waveform, sample_rate) ---
def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Loads audio, resamples, and converts to mono Tensor."""
    logger.info(f"Loading audio from: {audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != target_sr:
            logger.info(f"Resampling audio from {sample_rate} Hz to {target_sr} Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr

        # Convert to mono if necessary (average channels)
        if waveform.shape[0] > 1:
            logger.info(f"Converting audio from {waveform.shape[0]} channels to mono.")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        logger.info(f"Audio loaded. Shape: {waveform.shape}, Sample Rate: {sample_rate} Hz")
        return waveform, sample_rate

    except FileNotFoundError:
        logger.error(f"Audio file not found at {audio_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=True)
        return None, None

# --- Existing extract_speaker_audio_segments function ---
def extract_speaker_audio_segments(
    transcript_data: list,
    waveform: torch.Tensor,
    sample_rate: int,
    min_duration_sec: float = 2.0
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Extracts and concatenates audio segments for each speaker based on word timings.
    Filters out segments shorter than min_duration_sec.
    """
    logger.info(f"Starting extraction of speaker audio segments (Min Duration: {min_duration_sec}s, Target SR: {sample_rate}Hz).")
    speaker_segments = {} # {speaker_label: [list of tensors]}

    if waveform is None or sample_rate is None:
        logger.error("Waveform or sample rate is missing, cannot extract segments.")
        return {}, sample_rate

    logger.info("Extracting audio chunks based on transcript timings...")
    num_words = 0
    num_chunks = 0
    for segment in transcript_data:
        if 'words' in segment and isinstance(segment['words'], list):
            for word_info in segment['words']:
                num_words += 1
                start_time = word_info.get('start')
                end_time = word_info.get('end')
                speaker = word_info.get('speaker')

                if start_time is not None and end_time is not None and speaker is not None:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)

                    # Ensure indices are within bounds
                    if start_sample < end_sample <= waveform.shape[1]:
                        chunk = waveform[:, start_sample:end_sample]
                        if speaker not in speaker_segments:
                            speaker_segments[speaker] = []
                        speaker_segments[speaker].append(chunk)
                        num_chunks += 1
                    else:
                        logger.warning(f"Word '{word_info.get('word')}' time {start_time:.2f}-{end_time:.2f} out of bounds for waveform shape {waveform.shape}. Skipping chunk.")
                # else:
                #     logger.debug(f"Word '{word_info.get('word')}' missing start/end/speaker. Skipping chunk.")

    logger.info(f"Processed {num_words} words, extracted {num_chunks} initial audio chunks.")

    # Concatenate and filter by duration
    final_speaker_audio = {}
    logger.info(f"Filtering and concatenating segments based on min duration ({min_duration_sec}s)...")
    for speaker, chunks in speaker_segments.items():
        if not chunks: continue
        try:
            full_audio = torch.cat(chunks, dim=1)
            duration = full_audio.shape[1] / sample_rate
            if duration >= min_duration_sec:
                final_speaker_audio[speaker] = full_audio
                logger.info(f"  - Speaker {speaker}: Total duration {duration:.2f}s (Sufficient). Final shape: {full_audio.shape}")
            else:
                logger.info(f"  - Speaker {speaker}: Total duration {duration:.2f}s (Insufficient). Discarding.")
        except Exception as e:
            logger.error(f"Error concatenating chunks for speaker {speaker}: {e}", exc_info=True)


    logger.info(f"Audio segment extraction complete. Final segments for {len(final_speaker_audio)} speakers.")
    return final_speaker_audio, sample_rate


# --- NEW FUNCTION ---
def save_speaker_snippet(
    full_waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    max_snippet_sec: float,
    output_dir: str, # Specific directory for snippets, e.g., job_id/snippets
    filename_base: str # Base for filename, e.g., "snippet_UNKNOWN_1"
    ) -> Optional[str]:
    """
    Extracts a short audio snippet for a speaker and saves it as a WAV file.

    Args:
        full_waveform: The complete audio waveform tensor (mono).
        sample_rate: The sample rate of the waveform.
        start_sec: The start time of the speaker's segment.
        end_sec: The end time of the speaker's segment.
        max_snippet_sec: The maximum duration of the snippet to save.
        output_dir: The directory to save the snippet file.
        filename_base: The base name for the output file (e.g., "snippet_UNKNOWN_1").

    Returns:
        The full path to the saved snippet file, or None if saving failed.
    """
    if full_waveform is None or sample_rate is None:
        logger.error("Cannot save snippet: Waveform or sample rate is missing.")
        return None

    try:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # Calculate snippet duration and adjust end sample if needed
        max_snippet_samples = int(max_snippet_sec * sample_rate)
        snippet_end_sample = min(end_sample, start_sample + max_snippet_samples)

        # Ensure indices are valid
        if start_sample < 0 or snippet_end_sample > full_waveform.shape[1] or start_sample >= snippet_end_sample:
            logger.warning(f"Invalid sample range for snippet: {start_sample}-{snippet_end_sample} (Waveform shape: {full_waveform.shape}). Cannot save snippet for {filename_base}.")
            return None

        # Extract snippet tensor
        snippet_tensor = full_waveform[:, start_sample:snippet_end_sample]

        # Prepare filename and path
        snippet_filename = f"{filename_base}.wav" # Save as WAV
        output_path = os.path.join(output_dir, snippet_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save using soundfile (requires numpy array)
        # Convert tensor to numpy: [1, samples] -> [samples]
        snippet_numpy = snippet_tensor.squeeze().cpu().numpy()
        sf.write(output_path, snippet_numpy, sample_rate, subtype='PCM_16') # Save as 16-bit PCM WAV

        logger.info(f"Saved speaker snippet to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error saving snippet for {filename_base} to {output_dir}: {e}", exc_info=True)
        return None

