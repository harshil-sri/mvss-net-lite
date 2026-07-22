# Document Forgery Detection - Web App

FastAPI app with three parts: image + mask visualization, AI-generated report, and a chatbot scoped to one prediction.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000

## Enable real AI responses (optional)

By default, report/chat responses are stubbed placeholder text. To use real Claude API responses:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here   # set ANTHROPIC_API_KEY on Windows
```

No code changes needed — `llm_client.py` switches automatically when the key is present.

## Structure

```
app/
├── main.py              # app setup, mounts, router registration
├── routes/
│   ├── analyze.py       # POST /analyze
│   ├── report.py        # GET  /report/{id}
│   └── chat.py           # POST /chat/{id}
├── services/
│   ├── store.py          # in-memory prediction storage
│   ├── report_builder.py # mask overlay + confidence chart
│   └── llm_client.py     # stub/real LLM wrapper
├── templates/
│   ├── index.html
│   └── report.html
└── static/
    ├── uploads/
    └── generated/
```

## Replace the stub model

`routes/analyze.py` currently returns hardcoded prediction JSON. Swap that block for the real call to the detection model once it's ready — the JSON shape is already the contract the rest of the app expects.
# Backend
This directory is intended for the future backend API integration of the MVSS-Net Lite service. It is currently scaffolded and not yet implemented.
