Video Transcription (Azure SAS) — Streamlit UI + Flask API
This repo gives you two ways to run the same transcription pipeline:
•	Streamlit UI (app2.py): paste an Azure Blob SAS media URL → transcribe (Hindi → romanized “Hinglish” supported) → (optional) summary & questions → uploads outputs to your Azure container and shows the URLs.
•	Flask API (app.py): POST /process with an input Blob SAS + output Container SAS → returns only the transcript file URL (no summary URL), suitable for backend integration or automation.
The app uses OpenAI Whisper for transcription and handles long videos via chunking. Hindi output is normalized to romanized Hindi (Hinglish).
 
1) System Requirements
•	Python 3.10+
•	ffmpeg installed on the host
o	macOS: brew install ffmpeg
o	Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg
•	Internet access to download models the first time (Whisper/Transformers)
Apple Silicon (M1/M2/M3): works on CPU out of the box. For faster runs, use smaller Whisper models, greedy decoding, or adopt faster-whisper later.
 
2) Clone & Set Up Locally
# 1) Clone
git clone https://github.com/dibyacharyaAI/transcript_azure.git
cd transcript_azure

# 2) (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# .\venv\Scripts\activate     # Windows (PowerShell)

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
If you run into build issues, ensure Xcode CLT (mac) or build tools (Linux) are installed.
 
3) Azure SAS: What You Need
You need two SAS URLs:
1.	Input Blob SAS (for reading your media file):
o	Resource: Blob → sr=b
o	Permission: Read → sp=r
2.	Output Container SAS (for uploading results):
o	Resource: Container → sr=c
o	Permissions: Create + Write (minimum) → sp=cw
Recommended: Read + Create + Write + List → sp=rcwl
Time window:
•	st ≤ current time − 5 minutes (start)
•	se in the future (expiry)
•	Always use HTTPS.
If you see AuthorizationPermissionMismatch, regenerate the SAS with the correct resource and permissions.
 
4) Run the Streamlit UI
streamlit run app2.py
Using the UI
1.	Media URL: paste your Blob SAS URL (or any direct HTTP/HTTPS media URL).
2.	Container SAS URL: paste your Container SAS (this is where outputs are uploaded).
3.	Azure virtual folder: e.g., transcripts/ (files will be created under transcripts/<safe_title>/...).
4.	(Optional) Generate Questions: Off by default; turn on if needed. If it fails due to memory, the pipeline still completes (transcript+summary upload).
5.	(Optional) Upload original video to your output container.
When complete, the app displays copyable Azure URLs for:
•	transcription.txt
•	summary.txt
•	questions.txt (optional)
•	content_all.txt (combined)
You can also download these files locally from the UI.
 
5) Run the Flask API (Endpoint)
Start the server
python app.py
# or production-ready
# gunicorn -w 1 -k gthread -t 1200 wsgi:app
Server runs at http://localhost:8000 by default.
Health check
curl http://localhost:8000/health
API: POST /process
Request JSON
{
  "media_url": "https://<acct>.blob.core.windows.net/<in-container>/<file>.mp4?sv=...&sr=b&sp=r&sig=...",
  "output_container_sas_url": "https://<acct>.blob.core.windows.net/<out-container>?sv=...&sr=c&sp=rcwl&sig=...",
  "out_folder": "transcripts/",
  "generate_summary": false,
  "generate_questions": false,
  "upload_original": false
}
Response (success)
{
  "transcript_url": "https://<acct>.blob.core.windows.net/<out-container>/transcripts/<job>/transcription.txt?sv=...&sig=..."
}
cURL example
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
You can set AZURE_OUTPUT_CONTAINER_SAS_URL as an environment variable to avoid sending output_container_sas_url in the body.
 
6) Project Structure
.
├─ app2.py                 # Streamlit UI (Azure SAS in → uploads to Azure)
├─ app.py                  # Flask API (returns only transcript URL)
├─ wsgi.py                 # Gunicorn entrypoint for API
├─ modules/
│  ├─ transcription.py     # download/convert/preprocess/chunk + Whisper transcribe
│  ├─ hindi_support.py     # romanized Hindi cleanup
│  ├─ summarization.py     # Streamlit summary generation
│  ├─ question_generation.py# optional QG (kept resilient)
│  └─ utils.py             # YouTube helper for UI
├─ requirements.txt
└─ README.md
 
7) Deployment Notes (for the UI dev)
Streamlit (simple hosting)
•	Any VM/container that can run streamlit run app2.py
•	Make sure ffmpeg is installed
•	Provide users a secure way to input their Container SAS (don’t hardcode secrets in the repo)
Flask API (production)
•	Use gunicorn behind a reverse proxy (Nginx/Azure App Service/Container Apps)
•	Increase request timeout for long videos (-t 1200 in gunicorn example above)
•	Scale CPU/memory based on expected video length/throughput
•	Never log SAS tokens; avoid putting tokens in URLs in logs
Azure App Service / Container Apps
•	Build/run container with gunicorn wsgi:app
•	Expose port 8000
•	Configure environment variables (optional)
•	Review platform request timeouts for long-running sync requests
o	For very long videos, consider an async job pattern: POST /jobs → queue → worker processes → GET /jobs/{id} returns result URLs
 
8) Performance Tips
•	Model size: Use small / medium for faster runs vs large
•	Decoding: Greedy (beam_size=1, best_of=1, condition_on_previous_text=False) is fastest
•	I/O: Keep audio preprocessing single-pass; avoid multiple temp encodes
•	Long videos: processing time is roughly linear with duration
•	Apple Silicon: faster-whisper or whisper.cpp can be adopted later for more speed
 
9) Troubleshooting
•	AuthorizationPermissionMismatch → Output Container SAS missing c,w → regenerate with rcwlrecommended
•	FileNotFoundError: ffmpeg → Install ffmpeg on host
•	Question generation (on Apple GPU/MPS) crashes → QG is optional; keep OFF or run QG on CPU with a smaller T5 model
•	Streamlit widget duplication → handled (minimal UI widgets during transcription)
•	Very large files or slow network → increase timeouts (gunicorn -t 1200), or switch to async processing
 
10) Security & Git Hygiene
•	Never commit SAS tokens or secrets to Git
•	Use .gitignore:
•	venv/
•	__pycache__/
•	*.pyc
•	.streamlit/secrets.toml
•	*.mp4
•	*.wav
•	/tmp/
•	.DS_Store
•	Prefer fine-grained GitHub tokens with limited repo scope if you need CI/CD
 
11) Quick Run Checklist
•	 ffmpeg installed
•	 pip install -r requirements.txt
•	 Valid Blob SAS for input (sr=b, sp=r)
•	 Valid Container SAS for output (sr=c, sp=rcwl)
•	 Streamlit UI: run streamlit run app2.py
•	 API: run python app.py or gunicorn -w 1 -k gthread -t 1200 wsgi:app
 

