# app.py — Flask API aligned with the Streamlit Azure-SAS workflow
# Endpoint returns ONLY the transcript file URL (no summary URL),
# while keeping the same internal pipeline (language detect → transcribe → (optional) summary/QG).

import os
import re
import gc
import time
import tempfile
import logging
from typing import Optional
from urllib.parse import urlparse
from pathlib import Path

from flask import Flask, request, jsonify

# Azure SDK
from azure.storage.blob import BlobClient, ContainerClient, ContentSettings
from azure.core.exceptions import HttpResponseError

# Project modules (must be importable in PYTHONPATH)
from modules.transcription import transcribe_audio, detect_language_confidence
from modules.hindi_support import clean_hindi_text
from modules.summarization import generate_summary  # kept for parity (optional)
from modules.question_generation import generate_questions  # kept for parity (optional)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = Flask(__name__)

# ----------------- helpers -----------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-/]+", "_", (name or "")).strip("/")


def _ensure_container_sas(url: str):
    """Validate Container SAS: sr=c and has at least c,w perms (recommend rcwl)."""
    p = urlparse(url or "")
    if not p.scheme or not p.netloc or not p.query:
        raise ValueError("Invalid container SAS URL")
    q = p.query.lower()
    if "sr=c" not in q:
        raise ValueError("Container SAS must include sr=c")
    perms = dict(part.split("=", 1) for part in q.split("&") if "=" in part).get("sp", "")
    need = set(list("cw"))  # create + write
    if not need.issubset(set(perms)):
        raise ValueError("Container SAS needs at least create (c) and write (w); recommended rcwl")


def _download_to_temp(media_url: str) -> str:
    """Download Azure SAS blob or generic HTTP(S) to a local temp file and return the path."""
    try:
        if "blob.core.windows.net" in media_url and "sig=" in media_url:
            bc = BlobClient.from_blob_url(media_url)
            stream = bc.download_blob()
            data = stream.readall()
        else:
            import requests
            r = requests.get(media_url, stream=True, timeout=180)
            r.raise_for_status()
            data = b"".join(chunk for chunk in r.iter_content(chunk_size=1 << 20))
        ext = Path(urlparse(media_url).path).suffix or ".mp4"
        fh = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        fh.write(data); fh.flush(); fh.close()
        logger.info(f"Downloaded input -> {fh.name}")
        return fh.name
    except Exception as e:
        logger.exception("download failed")
        raise ValueError(f"Failed to fetch media: {e}")


def _upload_text(container_sas_url: str, blob_path: str, text: str) -> str:
    try:
        cc = ContainerClient.from_container_url(container_sas_url)
        bc = cc.get_blob_client(blob_path)
        bc.upload_blob(
            text.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain; charset=utf-8"),
        )
        return bc.url
    except HttpResponseError as e:
        logger.error(f"Azure upload error: {e}")
        raise
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        raise

# ----------------- routes -----------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
def process():
    """
    JSON body:
    {
      "media_url": "https://<acct>.blob.core.windows.net/<container>/<file>.mp4?sv=...&sr=b&sp=r&sig=...",
      "output_container_sas_url": "https://<acct>.blob.core.windows.net/<out-container>?sv=...&sr=c&sp=rcwl&sig=...",
      "out_folder": "transcripts/",                 # optional (default: transcripts/)
      "generate_summary": false,                      # optional (default: false)
      "generate_questions": false,                    # optional (default: false)
      "upload_original": false                        # optional (kept for parity, ignored in response)
    }

    Response:
      {"transcript_url": "https://.../transcripts/<job>/transcription.txt?..."}
    """
    try:
        body = request.get_json(silent=True) or {}
        media_url = (body.get("media_url") or "").strip()
        out_sas = (body.get("output_container_sas_url") or os.getenv("AZURE_OUTPUT_CONTAINER_SAS_URL", "")).strip()
        out_folder = _sanitize(body.get("out_folder") or "transcripts")
        gen_summary = bool(body.get("generate_summary", False))
        gen_questions = bool(body.get("generate_questions", False))
        upload_original = bool(body.get("upload_original", False))

        if not media_url or not media_url.startswith(("http://", "https://")):
            return jsonify({"error": "media_url must be a valid http(s) URL"}), 400
        if not out_sas:
            return jsonify({"error": "output_container_sas_url is required (Container SAS)"}), 400
        try:
            _ensure_container_sas(out_sas)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        base = Path(urlparse(media_url).path).stem or f"job_{int(time.time())}"
        base = _sanitize(base)
        run_folder = f"{out_folder}/{base}-{time.strftime('%Y%m%d-%H%M%S')}"

        local_media = None
        try:
            # 1) Download input to a local file
            local_media = _download_to_temp(media_url)

            # 2) Language detection
            lang_code, is_mixed, conf = detect_language_confidence(local_media)
            language = "english" if lang_code == "en" else ("hindi" if lang_code == "hi" else lang_code)
            logger.info(f"Language: {language} (confidence={conf:.2f})")

            # 3) Transcription (no UI widgets)
            text = transcribe_audio(local_media, transcription_area=None, language=language)
            if not text:
                return jsonify({"error": "Transcription failed or empty output"}), 500
            if language == "hindi":
                text = clean_hindi_text(text)

            # 4) (Optional) Summary/Questions — executed only if requested (not returned in API)
            if gen_summary:
                try:
                    _ = generate_summary(text, language=language)
                except Exception as e:
                    logger.warning(f"Summary generation skipped: {e}")
            if gen_questions:
                try:
                    _ = generate_questions(text)
                except Exception as e:
                    logger.warning(f"Question generation skipped: {e}")

            # 5) Upload ONLY the transcript, return its URL
            blob_path = f"{run_folder}/transcription.txt"
            transcript_url = _upload_text(out_sas, blob_path, text)

            # (Optional) upload original file for parity (not part of response)
            if upload_original:
                try:
                    cc = ContainerClient.from_container_url(out_sas)
                    bc = cc.get_blob_client(f"{run_folder}/original{Path(local_media).suffix or '.mp4'}")
                    with open(local_media, "rb") as f:
                        bc.upload_blob(f, overwrite=True)
                except Exception as e:
                    logger.warning(f"Original upload skipped: {e}")

            return jsonify({"transcript_url": transcript_url}), 200

        finally:
            try:
                if local_media and os.path.exists(local_media):
                    os.unlink(local_media)
            except Exception:
                pass
            gc.collect()

    except Exception as e:
        logger.exception("/process failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local dev
    # python app.py
    # Prod (wsgi.py expected): gunicorn -w 1 -k gthread -t 1200 wsgi:app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
