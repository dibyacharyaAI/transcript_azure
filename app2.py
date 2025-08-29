# app2.py  (Streamlit)
# Azure SAS input + Azure SAS output uploads + optional questions
import os
import re
import gc
import time
import tempfile
import traceback
import logging
from urllib.parse import urlparse
import streamlit as st
import torch

# Azure SDK
from azure.storage.blob import BlobClient, ContainerClient, ContentSettings
from azure.core.exceptions import HttpResponseError

# Project modules
from modules.transcription import (
    transcribe_audio,
    detect_language_confidence
)
from modules.utils import download_youtube_video
from modules.summarization import generate_summary
from modules.question_generation import generate_questions
from modules.hindi_support import clean_hindi_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app2")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Video Transcription & Summary (Azure SAS)",
    page_icon="🎬",
    layout="wide"
)
st.title("🎬 Video Transcription & Summary (Azure SAS)")

st.markdown(
    """
Is app me aap:
1) **Azure SAS Video URL** se seedha process kar sakte ho  
2) Local **Upload** ya **YouTube URL** bhi use kar sakte ho  
3) Output (**Transcription**, **Summary**, aur **Questions** optional) **Azure Container SAS** me upload hoga, aur last me uske **SAS URLs** milenge.

**Note:** Question generation optional hai. Agar error bhi aaye to pipeline rukegi nahi.
"""
)

# -----------------------------
# Utils (Azure + General)
# -----------------------------
def is_valid_youtube_url(url: str) -> bool:
    yt = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(yt, url or ""))

def guess_suffix_from_url(url: str, default=".mp4") -> str:
    path = urlparse(url).path.lower()
    for ext in (".mp4", ".mkv", ".mov", ".avi", ".mp3", ".wav", ".m4a"):
        if path.endswith(ext):
            return ext
    return default

def azure_download_blob_to_temp(blob_sas_url: str) -> str:
    """Download a blob (SAS URL) to a temp file and return local path."""
    try:
        bc = BlobClient.from_blob_url(blob_sas_url)
        data = bc.download_blob().readall()
        suffix = guess_suffix_from_url(blob_sas_url, ".mp4")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        logger.info(f"Downloaded Azure blob -> {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.error(f"Azure download failed: {e}")
        raise

def ensure_container_sas(url: str):
    """Basic check for container SAS (sr=c)."""
    if not url or "blob.core.windows.net" not in url:
        raise ValueError("Invalid Azure Container SAS URL.")
    q = urlparse(url).query.lower()
    if "sr=c" not in q:
        raise ValueError("Output SAS must be a **Container SAS** (sr=c).")
    if "sp=" not in q:
        raise ValueError("SAS must include permissions (sp=...).")
    # At minimum, need create+write for uploads
    perms = dict(part.split("=", 1) for part in q.split("&") if "=" in part).get("sp", "")
    needed = set(list("cw"))  # create + write
    if not needed.issubset(set(perms)):
        raise ValueError("Output Container SAS is missing permissions. Include at least Create (c) and Write (w). Recommended: r c w l.")
    return True

def azure_upload_text(container_sas_url: str, blob_name: str, text: str) -> str:
    """Upload text to container SAS and return blob SAS URL."""
    try:
        cc = ContainerClient.from_container_url(container_sas_url)
        bc = cc.get_blob_client(blob_name)
        bc.upload_blob(
            text.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain; charset=utf-8")
        )
        return bc.url
    except HttpResponseError as e:
        logger.error(f"Azure upload error: {e}")
        raise
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        raise

def azure_upload_file(container_sas_url: str, blob_name: str, local_path: str, content_type: str = None) -> str:
    try:
        cc = ContainerClient.from_container_url(container_sas_url)
        bc = cc.get_blob_client(blob_name)
        with open(local_path, "rb") as f:
            bc.upload_blob(
                f,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type) if content_type else None
            )
        return bc.url
    except HttpResponseError as e:
        logger.error(f"Azure upload error: {e}")
        raise
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        raise

def cleanup_paths(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

# -----------------------------
# Sidebar: Azure Output Settings
# -----------------------------
st.sidebar.header("⚙️ Output Settings (Azure)")
# Try secrets, else input
default_container_sas = ""
try:
    default_container_sas = st.secrets.get("AZURE_OUTPUT_CONTAINER_SAS_URL", "")
except Exception:
    pass

output_container_sas = st.sidebar.text_input(
    "Output Container SAS URL (sr=c, sp≥rcwl)",
    value=default_container_sas,
    type="password",
    help="Example: https://<account>.blob.core.windows.net/<container>?sv=...&sr=c&sp=rcwl&sig=..."
)

save_prefix = st.sidebar.text_input(
    "Output prefix (folder path inside container)",
    value="transcripts",
    help="Blobs will be created under this virtual folder."
)

upload_original_video = st.sidebar.checkbox("Also upload original video file to output container", value=False)

generate_questions_opt = st.sidebar.checkbox("Generate Questions (optional)", value=False)

st.sidebar.markdown(
    """
**Minimum SAS perms:**  
- **Input Blob SAS:** `sp=r`, `sr=b`  
- **Output Container SAS:** `sp=rcwl`, `sr=c`  
"""
)

# -----------------------------
# Core Processing
# -----------------------------
def run_pipeline(local_video_path: str, detected_language_hint: str = None):
    """Full pipeline: detect lang -> transcribe -> summary -> (optional) questions."""
    tmp_paths = [local_video_path]
    try:
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()

        # 1) Detect language
        with st.spinner("Detecting language..."):
            if detected_language_hint:
                lang_code = {"english": "en", "hindi": "hi"}.get(detected_language_hint.lower(), detected_language_hint)
                is_mixed = (lang_code == "hi-en")
                conf = 1.0
            else:
                lang_code, is_mixed, conf = detect_language_confidence(local_video_path)

            detected_language = "english" if lang_code == "en" else ("hindi" if lang_code == "hi" else lang_code)
            st.info(f"Detected language: **{detected_language.capitalize()}** (confidence: {conf:.2f})")

        # 2) Transcription (no UI widget flooding from modules to avoid duplicate keys)
        with st.spinner("Transcribing... (this may take a while)"):
            text = transcribe_audio(local_video_path, transcription_area=None, language=detected_language)
            if not text:
                raise RuntimeError("Transcription failed or empty result.")

            # For Hindi, ensure romanized (Hinglish) cleanup
            if detected_language == "hindi":
                text = clean_hindi_text(text)

        st.success("Transcription complete ✅")
        st.subheader("Transcription")
        st.text_area("Full Transcription", text, height=220, key="main_transcript_area")

        # 3) Summary
        with st.spinner("Generating summary..."):
            summary = generate_summary(text, language=detected_language)
        st.success("Summary ready ✅")
        st.subheader("Summary")
        st.write(summary)

        # 4) (Optional) Questions
        questions = []
        if generate_questions_opt:
            st.info("Generating questions (optional)...")
            try:
                # If your question generator struggles on MPS, you can force CPU by setting an env
                # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                questions = generate_questions(text)
                if not questions:
                    st.warning("No questions generated.")
                else:
                    st.subheader("Questions")
                    for i, q in enumerate(questions, 1):
                        st.write(f"{i}. {q}")
            except Exception as e:
                st.warning(f"Question generation skipped due to error: {e}")

        return detected_language, text, summary, questions
    finally:
        # memory clean
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()

def save_outputs_to_azure_or_fail(
    output_container_sas_url: str,
    base_name: str,
    transcription: str,
    summary: str,
    questions: list,
    original_local_path: str = None
):
    """Upload results to Azure container SAS. Raises if missing/invalid SAS."""
    ensure_container_sas(output_container_sas_url)  # validate

    ts = time.strftime("%Y%m%d-%H%M%S")
    base_folder = f"{save_prefix.strip('/')}/{base_name}-{ts}"

    urls = {}

    # Upload transcript
    t_blob = f"{base_folder}/transcript.txt"
    urls["transcript"] = azure_upload_text(output_container_sas_url, t_blob, transcription)

    # Upload summary
    s_blob = f"{base_folder}/summary.txt"
    urls["summary"] = azure_upload_text(output_container_sas_url, s_blob, summary)

    # Upload questions (only if exist and opted)
    if questions:
        q_blob = f"{base_folder}/questions.txt"
        q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        urls["questions"] = azure_upload_text(output_container_sas_url, q_blob, q_text)

    # Optionally upload original video
    if upload_original_video and original_local_path and os.path.exists(original_local_path):
        # Try to guess content-type
        ext = os.path.splitext(original_local_path)[1].lower()
        ctype = "video/mp4"
        if ext == ".mkv": ctype = "video/x-matroska"
        elif ext == ".mov": ctype = "video/quicktime"
        elif ext == ".avi": ctype = "video/x-msvideo"
        v_blob = f"{base_folder}/original{ext or '.mp4'}"
        urls["original_video"] = azure_upload_file(output_container_sas_url, v_blob, original_local_path, content_type=ctype)

    return urls

# -----------------------------
# Tabs (Inputs)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Azure SAS Video", "Upload Video", "YouTube URL"])

with tab1:
    st.header("Process from Azure **Blob SAS** URL")
    sas_url = st.text_input("Azure Blob SAS URL (video file)", value="", placeholder="https://<account>.blob.core.windows.net/<container>/<path>/<file>.mp4?sv=...&sr=b&sp=r&sig=...")
    st.caption("Note: This should be a **blob SAS** (sr=b) with **Read** permission. Example: a signed .mp4 in your container.")

    if st.button("Process Azure Video", type="primary", use_container_width=True):
        # Must have output container SAS to proceed (upload mandatory)
        if not output_container_sas:
            st.error("Output Container SAS required. Please paste a **Container SAS** with at least c,w permissions in the sidebar.")
        else:
            try:
                # Download the input blob to local temp (stable for all modules)
                local_path = azure_download_blob_to_temp(sas_url)
                base_name = os.path.splitext(os.path.basename(urlparse(sas_url).path))[0]

                lang, text, summary, questions = run_pipeline(local_path)

                # Upload results
                urls = save_outputs_to_azure_or_fail(
                    output_container_sas, base_name, text, summary, questions, original_local_path=local_path
                )

                st.success("All outputs uploaded to Azure ✅")
                st.markdown("**Uploaded URLs:**")
                for k, v in urls.items():
                    st.markdown(f"- **{k.capitalize()}**: {v}")

                # Also provide local downloads (optional convenience)
                st.download_button("Download Transcription (.txt)", data=text, file_name=f"{base_name}_transcript.txt")
                st.download_button("Download Summary (.txt)", data=summary, file_name=f"{base_name}_summary.txt")
                if questions:
                    q_txt = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                    st.download_button("Download Questions (.txt)", data=q_txt, file_name=f"{base_name}_questions.txt")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

with tab2:
    st.header("Upload a Video File")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "mkv", "mov", "avi"])
    if uploaded is not None:
        # Save to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".mp4")
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp.close()
        st.video(uploaded)

        if st.button("Process Uploaded Video", use_container_width=True):
            if not output_container_sas:
                st.error("Output Container SAS required. Please paste a **Container SAS** with at least c,w permissions in the sidebar.")
            else:
                try:
                    base_name = os.path.splitext(os.path.basename(uploaded.name))[0]
                    lang, text, summary, questions = run_pipeline(tmp.name)

                    urls = save_outputs_to_azure_or_fail(
                        output_container_sas, base_name, text, summary, questions, original_local_path=tmp.name
                    )
                    st.success("All outputs uploaded to Azure ✅")
                    st.markdown("**Uploaded URLs:**")
                    for k, v in urls.items():
                        st.markdown(f"- **{k.capitalize()}**: {v}")

                    st.download_button("Download Transcription (.txt)", data=text, file_name=f"{base_name}_transcript.txt")
                    st.download_button("Download Summary (.txt)", data=summary, file_name=f"{base_name}_summary.txt")
                    if questions:
                        q_txt = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                        st.download_button("Download Questions (.txt)", data=q_txt, file_name=f"{base_name}_questions.txt")

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())
                finally:
                    cleanup_paths([tmp.name])

with tab3:
    st.header("Enter YouTube URL")
    yt = st.text_input("YouTube URL")
    if yt and is_valid_youtube_url(yt):
        if st.button("Process YouTube Video", use_container_width=True):
            if not output_container_sas:
                st.error("Output Container SAS required. Please paste a **Container SAS** with at least c,w permissions in the sidebar.")
            else:
                local_path = None
                try:
                    with st.spinner("Downloading YouTube video..."):
                        local_path = download_youtube_video(yt)
                    st.success("YouTube video downloaded!")

                    base_name = os.path.splitext(os.path.basename(local_path))[0]
                    lang, text, summary, questions = run_pipeline(local_path)

                    urls = save_outputs_to_azure_or_fail(
                        output_container_sas, base_name, text, summary, questions, original_local_path=local_path
                    )
                    st.success("All outputs uploaded to Azure ✅")
                    st.markdown("**Uploaded URLs:**")
                    for k, v in urls.items():
                        st.markdown(f"- **{k.capitalize()}**: {v}")

                    st.download_button("Download Transcription (.txt)", data=text, file_name=f"{base_name}_transcript.txt")
                    st.download_button("Download Summary (.txt)", data=summary, file_name=f"{base_name}_summary.txt")
                    if questions:
                        q_txt = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                        st.download_button("Download Questions (.txt)", data=q_txt, file_name=f"{base_name}_questions.txt")

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())
                finally:
                    if local_path:
                        cleanup_paths([local_path])
    elif yt:
        st.error("Invalid YouTube URL.")
