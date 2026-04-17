# Resume ↔ Job Matcher

Universal skill extraction and matching for **all industries** — Software, Finance, Healthcare, MBA, Legal, Marketing, HR, and more.

Built with FastAPI, pluggable LLM (Ollama/Groq/OpenAI), Sentence Transformers, ChromaDB, and SQLite.

## Features

- **Candidate Portal** — Upload resumes (PDF/DOCX), extract skills via LLM, find matching jobs
- **Recruiter Portal** — Post jobs, extract requirements via LLM, find matching candidates
- **Jobs Board** — Browse all jobs, search by keyword, rank by candidate relevance
- **Google Sheets Sync** — Auto-import jobs from a Google Sheet, archive expired rows
- **Smart Skill Matching** — Synonym-aware, domain-aware fuzzy matching + semantic embeddings
- **Skills Taxonomy** — Automatically learns and tracks skills from all uploaded data
- **30-Day Retention** — Jobs older than 30 days are auto-cleaned from DB and archived in Google Sheets

## Prerequisites

1. **Python 3.10+**
2. **LLM Provider** — Choose one:
   - **Groq** (recommended for cloud, free tier): Get API key at [console.groq.com](https://console.groq.com)
   - **OpenAI**: Get API key at [platform.openai.com](https://platform.openai.com)
   - **Ollama** (local): Install from [ollama.ai](https://ollama.ai), then `ollama pull llama3.2`
3. **(Optional) Google Sheets API credentials** — Only needed if you want to sync jobs from a Google Sheet

## Setup

```bash
# Clone / navigate to the project
cd jobresume

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the App

### Using Groq (Cloud, Recommended)

Create a `.env` file:
```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile
```

Start the app:
```bash
uvicorn app:app --reload --port 8000
```

### Using OpenAI (Cloud)

Create a `.env` file:
```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk_your_openai_api_key
LLM_MODEL=gpt-4o-mini
```

### Using Ollama (Local)

1. Start Ollama in a separate terminal:
   ```bash
   ollama serve
   ```
   Verify the model is available:
   ```bash
   ollama list   # should show llama3.2
   ```

2. Create a `.env` file (or use defaults):
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

3. Start the app:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

4. **Open in browser**: [http://localhost:8000](http://localhost:8000)

## Pages

| URL | Description |
|---|---|
| `/` | Home — system status and navigation |
| `/candidate` | Upload resume, find matching jobs |
| `/recruiter` | Post a job, find matching candidates |
| `/jobs` | Browse/search all jobs, rank by candidate |
| `/sync_jobs` | Manually trigger Google Sheets sync |
| `/admin` | Dashboard — stats, skills taxonomy, recent data |

## Environment Variables

All optional — sensible defaults are built in.

### LLM Provider Configuration
| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `groq`, or `openai` |
| `LLM_API_KEY` | *(none)* | API key for Groq or OpenAI |
| `LLM_MODEL` | *(provider default)* | Override default model |
| `LLM_TIMEOUT` | `120` | LLM request timeout (seconds) |

### Ollama-specific (when LLM_PROVIDER=ollama)
| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | LLM model name |
| `OLLAMA_TIMEOUT` | `120` | LLM request timeout (seconds) |

### Database & Embedding
| Variable | Default | Description |
|---|---|---|
| `JOBRESUME_DB_PATH` | `jobresume.db` | SQLite database path |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB vector store path |
| `JOBRESUME_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `JOBRESUME_MAX_TEXT_CHARS` | `20000` | Max text chars to process |

### Google Sheets Integration (Optional)
| Variable | Default | Description |
|---|---|---|
| `GOOGLE_SHEET_ID` | *(none)* | Google Sheet ID for job sync |
| `GOOGLE_SHEETS_CREDENTIALS` | `google_credentials.json` | Path to service account JSON |
| `GOOGLE_SHEET_NAME` | `Sheet1` | Worksheet name to read from |
| `JOB_RETENTION_DAYS` | `30` | Days before jobs are auto-deleted |

## Google Sheets Integration (Optional)

To sync jobs from a Google Sheet:

1. Create a Google Cloud project and enable the **Google Sheets API** and **Google Drive API**
2. Create a **Service Account** and download the credentials JSON
3. Save it as `google_credentials.json` in the project root
4. Share your Google Sheet with the service account email (found in the JSON)
5. Set `GOOGLE_SHEET_ID` env var to your sheet's ID (from the sheet URL)

**Expected sheet columns**: `Job Title`, `Job Description`, `Posted Date`

Jobs older than 30 days are automatically moved to an `Archive_YYYY_MM` tab in the sheet and deleted from the database.

## Deploy to Render.com

Render.com provides a free tier for web services with automatic deployments from GitHub.

### Prerequisites
- GitHub account
- Render.com account (free tier works)

### Steps

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/jobresume.git
   git push -u origin main
   ```
   *(The `.gitignore` excludes `.env`, `chroma_db/`, `jobresume.db`, and other sensitive files)*

2. **Create a Web Service on Render**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click **New +** → **Web Service**
   - Connect your GitHub repository
   - Select the `jobresume` repo

3. **Configure Build & Start**
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables** (in Render dashboard → Environment section)
   ```
   LLM_PROVIDER=groq
   LLM_API_KEY=gsk_your_groq_api_key
   LLM_MODEL=llama-3.3-70b-versatile
   ```
   *(Optional: Add Google Sheets env vars if needed)*

5. **Deploy**
   - Click **Create Web Service**
   - Render will build and deploy automatically
   - Your app will be live at `https://your-app-name.onrender.com`

### Important Notes for Render
- Render's free tier spins down after 15 minutes of inactivity (cold start ~30s)
- The `chroma_db/` directory is stored in ephemeral storage — data persists within the same instance but may be lost if the instance is recreated
- For production persistence, consider using Render's Disk or an external database

## Project Structure

```
jobresume/
├── app.py                  # Main application (FastAPI)
├── requirements.txt        # Python dependencies
├── google_credentials.json # Google Sheets service account (not committed)
├── jobresume.db            # SQLite database (auto-created)
├── uploads/                # Uploaded resume files
└── README.md
```
