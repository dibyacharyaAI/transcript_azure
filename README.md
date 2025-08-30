cat > README.md << 'EOF'
# Video Transcription (Azure SAS)  Flask API

This repository provides two ways to run the same transcription pipeline:


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
    "media_url": "https://stgdhamkiit.blob.core.windows.net/lms-storage/transcripts/What%20is%20Artificial%20Intelligence%3F%20%7C%20Quick%20Learner.mp4?sp=r&st=2025-08-30T07:47:02Z&se=2025-08-30T16:02:02Z&sv=2024-11-04&sr=b&sig=UqnEXA99toj3Gxemxi9sXMZ%2BlNGJOH%2FDEVXOZFxocFQ%3D",
    "output_container_sas_url": "https://stgdhamkiit.blob.core.windows.net/lms-storage?sp=rcw&st=2025-08-30T07:52:14Z&se=2025-08-30T16:07:14Z&sv=2024-11-04&sr=c&sig=1WNfoFrMNu1VuTzOCft0NOJD9NNaE9FF3y7nbRDUaxU%3D",
    "out_folder": "transcripts/",
    "generate_summary": false,
    "generate_questions": false,
    "upload_original": false
  }'


Success response:

{"transcript_url":"https://stgdhamkiit.blob.core.windows.net/lms-storage/transcripts/What_20is_20Artificial_20Intelligence_3F_20_7C_20Quick_20Learner-20250830-145838/transcription.txt?sp=rcw&st=2025-08-30T07%3A52%3A14Z&se=2025-08-30T16%3A07%3A14Z&sv=2024-11-04&sr=c&sig=1WNfoFrMNu1VuTzOCft0NOJD9NNaE9FF3y7nbRDUaxU%3D"}



```

Optional environment variable:
If you set AZURE_OUTPUT_CONTAINER_SAS_URL, you can omit output_container_sas_url in the body.

6) Project Structure
```bash
.
               
├─ app.py                  # Flask API (returns only transcript URL)
├─ wsgi.py                 # Gunicorn entrypoint for the API
├─ modules/
│  ├─ transcription.py     # download/convert/preprocess/chunk + Whisper transcribe
│  ├─ hindi_support.py     # romanized Hindi cleanup
│  ├─ summarization.py     # summary generation
│  ├─ question_generation.py# optional QG (resilient; can be disabled)
│  └─ utils.py             # YouTube helper for the UI
├─ requirements_app.txt
└─ README.md
```



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

