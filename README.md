# GenAI Subsidiary Extraction Solution

This project extracts subsidiary information for parent companies from public web sources, annual reports, PDFs, and other web pages. It provides:

- A **FastAPI backend** for Excel upload, background extraction, result review, download, and chatbot Q&A.
- A **Streamlit dashboard** for uploading company Excel files, monitoring jobs, viewing results, manual review, and asking questions.
- **SQLite** for audit and result storage.
- **ChromaDB** for persistent vector storage.
- **Ollama** for LLM-based subsidiary extraction and Q&A.
- **Tesseract OCR** for scanned PDF pages.

## Project Structure

```text
subsidiary_extraction_solution/
├── app.py                                  # FastAPI backend
├── streamlit_app.py                        # Streamlit dashboard
├── requirements.txt                        # Python dependencies
├── Dockerfile                              # Docker image definition
├── .dockerignore                           # Docker build exclusions
├── subsidiary_audit.db                     # SQLite database, created/updated at runtime
├── vector_db/                              # ChromaDB persistent storage
├── input_*.xlsx                            # Uploaded/input Excel files
└── output_*.xlsx                           # Generated output files
```

## Main Features

1. Upload Excel file containing company names.
2. Search public web sources using DuckDuckGo search.
3. Fetch HTML/PDF content from ranked sources.
4. Extract PDF text using PyMuPDF and Tesseract OCR when needed.
5. Store source chunks in ChromaDB.
6. Use SentenceTransformer embeddings and CrossEncoder reranking.
7. Use Ollama model to extract subsidiaries as strict JSON.
8. Store results and source audit records in SQLite.
9. Review/correct extracted rows through Streamlit.
10. Ask Q&A questions about processed subsidiary data.

## Requirements

### Local Requirements

- Python 3.11+
- Ollama installed and running
- Required Ollama model pulled locally, for example:

```bash
ollama pull mistral:latest
```

- Tesseract OCR installed if running locally without Docker

## Python Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: for FastAPI file upload support, `python-multipart` is required. If it is not already installed, run:

```bash
pip install python-multipart
```

## Run FastAPI Backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API documentation:

```text
http://localhost:8000/docs
```

## Run Streamlit Dashboard

In another terminal:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Open:

```text
http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload Excel and start extraction job |
| GET | `/job/{job_id}` | Check job status |
| GET | `/results/{job_id}` | Get extracted subsidiary results |
| GET | `/sources/{job_id}` | Get source audit records |
| PUT | `/review/{result_id}` | Update/correct reviewed result |
| GET | `/download/{job_id}` | Download output Excel |
| POST | `/chat` | Ask Q&A over extracted company data |

## Input Excel Format

The application automatically detects common column names.

Recommended columns:

```text
Parent Company Name
Parent Company Country
Website
```

Other supported company column names include:

```text
Company name
Company Name
company
Company
Parent Company Enterprise
parent_company
```

Other supported country column names include:

```text
Country
country
Location
location
```

Other supported website column names include:

```text
Website
website
Domain
domain
URL
url
```

## Docker Build

Build the image:

```bash
docker build -t subsidiary-extraction-solution .
```

## Docker Run - FastAPI

If Ollama is running on your host machine, run:

```bash
docker run --rm -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=mistral:latest \
  -v $(pwd)/data:/app/data \
  subsidiary-extraction-solution
```

On Linux, `host.docker.internal` may need this extra option:

```bash
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=mistral:latest \
  -v $(pwd)/data:/app/data \
  subsidiary-extraction-solution
```

Then open:

```text
http://localhost:8000/docs
```

## Docker Run - Streamlit

The Dockerfile defaults to FastAPI. To run Streamlit using the same image:

```bash
docker run --rm -p 8501:8501 \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  subsidiary-extraction-solution \
  streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

> Important: `streamlit_app.py` currently uses `API_BASE_URL = "http://127.0.0.1:8000"`. If Streamlit runs in a separate container, update it to read from environment variable or set the correct backend URL.

Suggested change:

```python
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral:latest` | Ollama model used for extraction and chatbot |
| `OLLAMA_HOST` | Ollama client default | Ollama server URL, for example `http://host.docker.internal:11434` |

## Runtime Data

The application creates or updates these files/directories:

```text
subsidiary_audit.db
vector_db/
input_<job_id>_<filename>.xlsx
output_<job_id>.xlsx
```

For Docker usage, mount a volume if you want to persist runtime data.

## Troubleshooting

### FastAPI upload error: python-multipart required

Install:

```bash
pip install python-multipart
```

### Ollama connection error

Confirm Ollama is running:

```bash
ollama serve
```

Confirm the model exists:

```bash
ollama list
```

Pull the model if missing:

```bash
ollama pull mistral:latest
```

### Docker container cannot connect to Ollama

Use:

```bash
-e OLLAMA_HOST=http://host.docker.internal:11434
```

On Linux, add:

```bash
--add-host=host.docker.internal:host-gateway
```

### OCR does not work

Ensure Tesseract is installed. The Dockerfile installs it automatically.

### Slow first startup

`sentence-transformers` and `cross-encoder` models may download on first run. Ensure internet access is available or pre-cache the models.

## Notes

- Extraction quality depends on the availability and quality of public source documents.
- The LLM is instructed not to hallucinate and to return `Not available` when data is missing.
- Manual review is recommended before using the results in production.
