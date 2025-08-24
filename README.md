# Simple AI-Powered Legal Assistant

Simple, local-first AI-powered legal document generator with FastAPI + Gradio, streaming, and DOCX/PDF export.

## Features
- Shared async Ollama client (`services/ollama_client.py`) with timeouts and error handling
- Legal generator service (`services/legal_generator.py`) with normalized document types and templates
- FastAPI API (`app.py`) with request/response models, health endpoint, CORS, and mounted Gradio UI at `/ui`
- Gradio UI (`legal_assistant.py`) with conditional fields and model parameter controls
- Environment-driven config via `config.py` (`.env` supported)

## Requirements
- Python 3.10+
- Ollama running locally and serving the target model

Example: start Ollama and pull model
```bash
ollama run deepseek-r1
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Optional `.env` file:
```
OLLAMA_URL=http://localhost:11434/api/generate
MODEL_NAME=deepseek-r1
REQUEST_TIMEOUT=60
TEMPERATURE=0.3
TOP_P=0.9
NUM_PREDICT=512
```

## Run
- API + UI:
```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs for API docs
Open http://127.0.0.1:8000/ui for the Gradio UI

- UI standalone:
```bash
python legal_assistant.py
```

## API
POST `/legal/`
```json
{
  "doc_type": "rental agreement",
  "party1": "Alice",
  "party2": "Bob",
  "duration": "12",
  "salary": "",
  "temperature": 0.3,
  "top_p": 0.9,
  "num_predict": 512
}
```
Response:
```json
{ "response": "...generated text..." }
```

## Notes
- This app generates AI-drafted documents and must be reviewed by a qualified attorney.
- Consider enabling auth and rate limits before exposing publicly.
