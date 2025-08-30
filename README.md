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
pip install -r requirements_app.txt


```
3) Azure SAS Requirements
You need two SAS URLs:
Input Blob SAS (to read your media file)
Resource: Blob (sr=b)
Permission: Read (sp=r)
Output Container SAS (to upload results)
Resource: Container (sr=c)
Permissions: at least Create and Write (sp=cw)
Recommended: Read, Create, Write, List (sp=rcwl)
Time window:
st (start) should be <= current time minus ~5 minutes (to avoid clock skew).
se (expiry) should be in the future.
Always use HTTPS.
If you see AuthorizationPermissionMismatch, regenerate the SAS with the correct resource and permissions.


4) Run the Streamlit UI
```bash
streamlit run app2.py
```
Using the UI:
Media URL: paste your Blob SAS URL (or any direct HTTP/HTTPS media URL).
Container SAS URL: paste your Container SAS (this is where outputs are uploaded).
Azure virtual folder: e.g., transcripts/ (files will be created under transcripts/<safe_title>/...).
Optional: Generate Questions (default OFF). If it fails due to memory limits, the pipeline still completes for transcript and summary.
Optional: Upload original video file to your output container.
Upon completion, the app displays copyable Azure URLs for:
transcription.txt
summary.txt
questions.txt (optional)
content_all.txt (combined)
Local download buttons are also provided.


5) Run the Flask API (Endpoint)
Start the server:

```bash
python app.py
# or production:
gunicorn -w 1 -k gthread -t 1200 wsgi:app

```
The server runs at http://localhost:8000 by default.
```bash
curl http://localhost:8000/health
```
POST /process:
Request JSON:

```bash
{
  "media_url": "https://<acct>.blob.core.windows.net/<in-container>/<file>.mp4?sv=...&sr=b&sp=r&sig=...",
  "output_container_sas_url": "https://<acct>.blob.core.windows.net/<out-container>?sv=...&sr=c&sp=rcwl&sig=...",
  "out_folder": "transcripts/",
  "generate_summary": false,
  "generate_questions": false,
  "upload_original": false
}
```

Success response:
```bash
{
  "transcript_url": "https://<acct>.blob.core.windows.net/<out-container>/transcripts/<job>/transcription.txt?sv=...&sig=..."
}
```

cURL example:
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://<acct>.blob.core.windows.net/<in-container>/<file>.mp4?sv=...&sr=b&sp=r&sig=...",
    "output_container_sas_url": "https://<acct>.blob.core.windows.net/<out-container>?sv=...&sr=c&sp=rcwl&sig=...",
    "out_folder": "transcripts/",
    "generate_summary": false,
    "generate_questions": false,
    "upload_original": false
  }'

```

Optional environment variable:
If you set AZURE_OUTPUT_CONTAINER_SAS_URL, you can omit output_container_sas_url in the body.

6) Project Structure
```bash
.
├─ app2.py                 # Streamlit UI (Azure SAS input → uploads to Azure)
├─ app.py                  # Flask API (returns only transcript URL)
├─ wsgi.py                 # Gunicorn entrypoint for the API
├─ modules/
│  ├─ transcription.py     # download/convert/preprocess/chunk + Whisper transcribe
│  ├─ hindi_support.py     # romanized Hindi cleanup
│  ├─ summarization.py     # Streamlit summary generation
│  ├─ question_generation.py# optional QG (resilient; can be disabled)
│  └─ utils.py             # YouTube helper for the UI
├─ requirements.txt
└─ README.md
```

8) Deployment Notes
Streamlit (simple):
Any VM/container that can run streamlit run app2.py.
Ensure ffmpeg is installed.
Provide a secure way for users to input the Container SAS (do not hardcode secrets).
Flask API (production):
Use gunicorn behind a reverse proxy (Nginx, Azure App Service, or Azure Container Apps).
Increase timeouts for long videos (e.g., gunicorn -t 1200).
Scale CPU/RAM based on expected video duration and traffic.
Avoid logging SAS tokens or embedding them in logs.
Azure App Service / Container Apps:
Build and run with gunicorn wsgi:app on port 8000.
Configure environment variables as needed.
Be mindful of platform request timeouts for long-running requests.
For very long videos, consider an async job pattern:
POST /jobs returns a job ID
A worker processes the job in the background
GET /jobs/{id} returns status and result URLs

9) Performance Tips
Prefer smaller models (small/medium) over large for speed.
Use greedy decoding (beam_size=1, best_of=1, condition_on_previous_text=False).
Keep audio preprocessing single-pass to reduce I/O.
Processing time is roughly linear in audio duration.
On Apple Silicon, consider faster-whisper or whisper.cpp for further speed (optional).

10) Troubleshooting
AuthorizationPermissionMismatch: Output Container SAS is missing required permissions. Regenerate with at least sp=cw (recommended sp=rcwl).
FileNotFoundError: ffmpeg: Install ffmpeg on the host.
Question generation crashes (Apple MPS/GPU): Questions are optional; keep OFF or run QG on CPU with a smaller T5 model.
Streamlit duplicate widget errors: Addressed by minimizing UI widgets during transcription.
Very large files/slow networks: Increase timeouts (gunicorn -t 1200) or adopt async processing.

