cat > README.md << 'EOF'
# Video Transcription (Azure SAS) — Streamlit UI + Flask API

This repository provides two ways to run the same transcription pipeline:

- Streamlit UI (`app2.py`): Paste an Azure Blob SAS media URL, transcribe (Hindi is normalized to romanized "Hinglish"), optionally generate a summary and questions, and upload outputs to your Azure container. The app shows the resulting URLs.
- Flask API (`app.py`): `POST /process` with an input Blob SAS and an output Container SAS. The API returns only the transcript file URL (no summary URL), suitable for backend integration.

The pipeline uses OpenAI Whisper and processes long audio via chunking.

---

## 1) System Requirements

- Python 3.10+
- ffmpeg installed on the host
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Internet access for first-time model downloads

Apple Silicon (M1/M2/M3) works on CPU by default. For faster runs, prefer smaller Whisper models and greedy decoding, or consider faster-whisper/whisper.cpp later (optional).

---

## 2) Clone and Set Up Locally

```bash
# 1) Clone
git clone https://github.com/dibyacharyaAI/transcript_azure.git
cd transcript_azure

# 2) Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# .\venv\Scripts\activate     # Windows (PowerShell)

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

