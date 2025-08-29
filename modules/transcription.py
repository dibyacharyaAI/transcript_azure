import whisper
import streamlit as st
import logging
import os
import torch
import re
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import gc
from functools import lru_cache
from typing import Tuple, Dict, Optional

# --- NEW: URL / Azure helpers ---
import tempfile
from urllib.parse import urlparse, unquote

try:
    # Optional; we'll gracefully fallback to plain HTTP if missing
    from azure.storage.blob import BlobClient  # type: ignore
    _HAS_AZURE = True
except Exception:
    _HAS_AZURE = False

import requests

logger = logging.getLogger(__name__)

HINDI_PROMPT = """
This is a transcription of Hindi content. Please transcribe in romanized Hindi (Hinglish) instead of Devanagari script.
Examples:
- "मैं आज बहुत खुश हूँ" should be written as "Main aaj bahut khush hoon"
- "यह एक अच्छा वीडियो है" should be written as "Yeh important topic hai"
- "आपका स्वागत है" should be written as "Aapka swagat hai"
IMPORTANT: Use ONLY Roman script (English alphabets) for ALL Hindi words. NEVER use Devanagari script in the output.
Ensure natural spelling in romanized form and maintain the meaning of the original content.
"""

HINGLISH_PROMPT = """
This is a transcription of mixed Hindi-English content that may include:
- Technical discussions
- Educational lectures
- Professional conversations
Please maintain natural code-switching between Hindi and English.
Output all Hindi words in romanized form (using Roman script) rather than Devanagari.
Examples:
- "मैं computer use करता हूँ" should be written as "Main computer use karta hoon"
- "यह important topic है" should be written as "Yeh important topic hai"
- "क्या आप marketing strategy समझा सकते हैं" should be written as "Kya aap marketing strategy samjha sakte hain"
IMPORTANT: Use ONLY Roman script (English alphabets) for ALL Hindi words. NEVER use Devanagari script in the output.
Ensure the transcription maintains the mixed language nature with English technical terms preserved.
"""

# ===============================
# Azure / URL handling utilities
# ===============================

def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _looks_like_azure_sas(url: str) -> bool:
    try:
        p = urlparse(url)
        # basic check for Azure blob and SAS params
        is_blob = p.netloc.endswith(".blob.core.windows.net")
        has_sas = ("sig=" in (p.query or "")) or ("sv=" in (p.query or ""))
        return is_blob and has_sas
    except Exception:
        return False


def _download_via_requests(url: str, suffix: Optional[str] = None) -> str:
    """Streaming download using requests (works for SAS too)."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    # derive suffix from path
    if suffix is None:
        path = urlparse(url).path
        suffix = Path(path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    logger.info(f"Downloaded via HTTP -> {tmp.name}")
    return tmp.name


def _download_azure_sas(url: str) -> str:
    """Prefer Azure SDK if available, else fallback to requests."""
    if _HAS_AZURE:
        try:
            bc = BlobClient.from_blob_url(url)
            props = bc.get_blob_properties()
            # guess extension from content-type or url path
            content_type = (props.content_settings.content_type or "").lower()
            ext = ".mp4"
            if "audio" in content_type and "/wav" in content_type:
                ext = ".wav"
            elif "audio" in content_type and "/mpeg" in content_type:
                ext = ".mp3"
            elif "video" in content_type and "/quicktime" in content_type:
                ext = ".mov"
            else:
                # fallback from path
                path = urlparse(url).path
                ext = Path(path).suffix or ext

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            with open(tmp.name, "wb") as f:
                stream = bc.download_blob()
                for c in stream.chunks():
                    f.write(c)
            logger.info(f"Downloaded via Azure SDK -> {tmp.name}")
            return tmp.name
        except Exception as e:
            logger.warning(f"Azure SDK download failed, falling back to HTTP: {e}")
    # fallback
    return _download_via_requests(url)


def resolve_input_source(path_or_url: str) -> str:
    """Return a local filesystem path for either a local path or an Azure SAS/HTTP URL."""
    if not path_or_url:
        raise ValueError("Empty audio/video path provided")

    # local file
    if os.path.exists(path_or_url):
        return path_or_url

    # URL cases
    if _is_url(path_or_url):
        if _looks_like_azure_sas(path_or_url):
            return _download_azure_sas(path_or_url)
        # generic http(s)
        return _download_via_requests(path_or_url)

    # otherwise, return as-is (ffmpeg may handle some schemes)
    return path_or_url


# ===============================
# Whisper helpers (original logic)
# ===============================

@lru_cache(maxsize=1)
def get_whisper_model(model_name: str = "large", device: str = "cpu"):
    """Cache the model to prevent repeated loading."""
    logger.info(f"Loading Whisper {model_name} model on {device}")
    try:
        model = whisper.load_model(model_name)
        model = model.to(device)
        return model
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        raise


def load_audio_chunk(file_path: str, chunk_size: int = 480000, offset: int = 0):
    """Load a chunk of audio with proper memory handling (expects a LOCAL path)."""
    try:
        # Try whisper first (calls ffmpeg under the hood)
        try:
            audio = whisper.load_audio(file_path)
        except Exception as whisper_error:
            logger.warning(f"Whisper failed to load audio: {str(whisper_error)}")
            import librosa
            logger.info("Trying to load audio with librosa...")
            audio, _ = librosa.load(file_path, sr=16000, mono=True)

        if offset >= len(audio):
            return None
        end = min(offset + chunk_size, len(audio))
        return audio[offset:end]
    except Exception as e:
        logger.error(f"Error loading audio chunk: {str(e)}")
        return None


def detect_language_confidence(audio_path: str) -> Tuple[str, bool, float]:
    """Detect language with confidence score, with improved English detection.
    Accepts local path or URL (Azure SAS supported). Returns (lang_code, is_mixed, confidence).
    """
    try:
        local_path = resolve_input_source(audio_path)
        # Load ~30s chunk for detection (30*16000)
        audio = load_audio_chunk(local_path, chunk_size=480000)
        if audio is None:
            logger.warning("Could not load audio for language detection, using English as fallback")
            return "en", False, 0.8

        audio = whisper.pad_or_trim(audio)
        # Use a smaller model for more robust detection and fewer shape surprises
        model = get_whisper_model("small", device="cpu")

        # Build mel spectrogram
        try:
            mel = whisper.log_mel_spectrogram(audio).to("cpu")
            # ensure batch dimension
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            # Whisper reference models expect 80 mel bins
            expected_channels = 80
            if mel.shape[1] != expected_channels:
                logger.warning(
                    f"Mel spectrogram has {mel.shape[1]} channels instead of expected {expected_channels}; fallback to EN"
                )
                return "en", False, 0.75
        except Exception as mel_error:
            logger.warning(f"Error generating mel spectrogram: {str(mel_error)}")
            logger.info("Using English as fallback language")
            return "en", False, 0.75

        # Detect with timeout
        import threading

        result = {"success": False, "langs": None}

        def _detect():
            try:
                _, probs = model.detect_language(mel)
                result["success"] = True
                result["langs"] = probs
            except Exception as detect_error:
                logger.error(f"Language detection internal error: {str(detect_error)}")

        t = threading.Thread(target=_detect)
        t.start()
        t.join(timeout=10)

        if not result["success"]:
            logger.warning("Language detection timed out or failed, using English as fallback")
            return "en", False, 0.7

        probs = result["langs"]
        top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]

        # Prioritize English if reasonably strong
        if any(lang == "en" and conf > 0.35 for lang, conf in top_langs):
            en_conf = next(conf for lang, conf in top_langs if lang == "en")
            logger.info(f"Prioritizing English detection with confidence: {en_conf:.2f}")
            return "en", False, float(en_conf)

        # Mixed Hindi-English heuristic
        if len(top_langs) == 2 and {top_langs[0][0], top_langs[1][0]} <= {"hi", "en"}:
            if abs(top_langs[0][1] - top_langs[1][1]) < 0.3:
                logger.info("Detected potential code-mixing between Hindi and English")
                return top_langs[0][0], True, float(top_langs[0][1])

        logger.info(f"Detected language: {top_langs[0][0]} with confidence: {top_langs[0][1]:.2f}")
        return top_langs[0][0], False, float(top_langs[0][1])

    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        logger.info("Falling back to English language detection")
        return "en", False, 0.8


def process_audio_chunk(chunk, model, language, is_mixed, prompt=""):
    """Process a single audio chunk."""
    try:
        chunk = whisper.pad_or_trim(chunk)
        transcription_options = {
            "language": language,
            "initial_prompt": prompt,
            "condition_on_previous_text": True,
            "fp16": False,
        }
        if language == "en":
            transcription_options.update({
                "temperature": 0,
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "suppress_tokens": [-1],
            })
        result = model.transcribe(chunk, **transcription_options)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""


def preprocess_audio_for_english(audio_path: str) -> str:
    """Apply English-specific audio preprocessing for better transcription results.
    Returns a path (possibly a temp wav) to be used for transcription.
    """
    try:
        if not os.path.isfile(audio_path):
            logger.error(f"File not found: {audio_path}")
            return audio_path

        import subprocess
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()

        try:
            cmd = ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", temp_wav_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Converted audio file to WAV format at {temp_wav_path}")
            audio_data, sample_rate = sf.read(temp_wav_path)
        except Exception as ffmpeg_error:
            logger.warning(f"Error converting file with ffmpeg: {str(ffmpeg_error)}")
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
                logger.info("Successfully loaded audio with librosa")
            except Exception as librosa_error:
                logger.error(f"Error loading audio with librosa: {str(librosa_error)}")
                return audio_path

        # stereo -> mono (safety), normalize, light highpass
        if hasattr(audio_data, "shape") and len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

        from scipy import signal
        sos = signal.butter(10, 80, 'hp', fs=sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, audio_data)

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(out.name, filtered_audio, sample_rate)
        logger.info(f"Applied English audio preprocessing -> {out.name}")
        return out.name
    except Exception as e:
        logger.error(f"Error in English audio preprocessing: {str(e)}")
        logger.info("Using original audio file")
        return audio_path


def transcribe_audio(
    audio_path: str,
    transcription_area=None,
    language: Optional[str] = None,
    progress_callback=None,
    **kwargs
):
    """Transcribe audio with improved memory handling and error recovery.
    Accepts local path or URL (including Azure Blob SAS). All other logic preserved.
    """
    start_time = time.time()
    session_key = f"tx_{int(time.time()*1e6)}"

    is_mixed = False
    confidence = 0.0

    try:
        # 1) Resolve remote (Azure SAS / HTTP) to local temp file
        local_path = resolve_input_source(audio_path)

        # 2) Language detect (or respect provided)
        if language is None:
            language, is_mixed, confidence = detect_language_confidence(local_path)
            logger.info(f"Detected language: {language} (mixed: {is_mixed}, confidence: {confidence:.2f})")
        else:
            if language.lower() == "english":
                language = "en"; confidence = 1.0
            elif language.lower() == "hindi":
                language = "hi"; confidence = 1.0
            is_mixed = (language == "hi-en")
            if is_mixed:
                language = "hi"

        # 3) Optional English-specific preprocessing
        processed_audio_path = local_path
        if language == "en" and not is_mixed:
            if transcription_area:
                preprocess_status = transcription_area.empty()
                preprocess_status.info("Preprocessing audio for enhanced English transcription…")
            processed_audio_path = preprocess_audio_for_english(local_path)

        # 4) UI: language info
        progress_bar = None
        progress_text = None
        output_area = None
        if transcription_area:
            status = transcription_area.empty()
            lang_display = "English" if language == "en" else ("Hinglish" if is_mixed or language == "hi" else language.upper())
            status.info(f"Detected {lang_display} content (Confidence: {confidence:.2f})")
            progress_bar = transcription_area.progress(0)
            progress_text = transcription_area.empty()
            output_area = transcription_area.empty()

        # 5) Load model
        model = get_whisper_model("large", device="cpu")

        # 6) Prompt selection for romanization
        if is_mixed:
            prompt = HINGLISH_PROMPT
            logger.info("Using Hinglish prompt for mixed Hindi-English content")
        elif language == "hi":
            prompt = HINDI_PROMPT
            logger.info("Using Hindi prompt with romanized output instructions")
        else:
            prompt = ""
            logger.info(f"No special prompt for language: {language}")

        # 7) Chunked processing
        chunk_size = 480000  # ~30s @ 16kHz
        offset = 0
        transcription_parts: list[str] = []
        chunk_count = 0

        while True:
            chunk = load_audio_chunk(processed_audio_path, chunk_size, offset)
            if chunk is None or len(chunk) == 0:
                break

            chunk_text = process_audio_chunk(chunk, model, language, is_mixed, prompt)
            if chunk_text:
                if language == "hi" and re.search(r"[\u0900-\u097F]", chunk_text):
                    logger.warning("Detected Devanagari despite romanization prompt. Cleaning…")
                    from modules.hindi_support import clean_hindi_text
                    chunk_text = clean_hindi_text(chunk_text)
                transcription_parts.append(chunk_text)

            # UI progress (approximate)
            progress_val = min(1.0, (chunk_count + 1) * 0.1)
            if progress_bar is not None:
                progress_bar.progress(progress_val)
                if progress_text is not None:
                    progress_text.text(f"Processing chunk {chunk_count + 1}")
                if output_area is not None:
                    output_area.text_area(
                        "Current transcription",
                        " ".join(transcription_parts),
                        height=200,
                        key=f"current_tx_{session_key}_{chunk_count}",
                    )

            # external progress callback (optional)
            cb = progress_callback or kwargs.get("on_progress")
            if cb:
                try:
                    cb(progress_val, f"chunk {chunk_count + 1}")
                except Exception:
                    pass

            offset += chunk_size
            chunk_count += 1
            gc.collect()
            if hasattr(torch, "cuda"):
                torch.cuda.empty_cache()

        # 8) Finalize text
        final_text = " ".join(transcription_parts)
        if language == "hi":
            from modules.hindi_support import clean_hindi_text
            final_text = clean_hindi_text(final_text)

        total_time = time.time() - start_time
        duration = offset / 16000.0
        speed_factor = (duration / total_time) if total_time > 0 else 0.0
        logger.info(f"Transcription completed in {total_time:.2f}s ({speed_factor:.2f}x real-time)")

        if progress_bar is not None:
            progress_bar.progress(1.0)
            if progress_text is not None:
                progress_text.text("Transcription complete!")
            if output_area is not None:
                output_area.text_area(
                    "Final Transcription",
                    final_text,
                    height=300,
                    key=f"final_tx_{session_key}",
                )

        return final_text

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        if transcription_area:
            transcription_area.error(f"Error: {str(e)}")
        return None
