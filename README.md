# DocAI — Document Intelligence Platform

> Upload any PDF, DOCX, or TXT. Ask questions, extract structured data, and compare documents — powered by Groq + LangChain RAG. 100% FREE.

\---

## What You Can Do

* **Q\&A** — Ask natural language questions across any document
* **Extract** — Pull invoice fields, contract terms, resume data as clean JSON
* **Compare** — Find similarities and differences between two documents

\---



\## Screenshots



\### Ask Questions with Source Citations

!\[Q\&A Demo](screenshots/qa-demo.png)



\### Extract Structured Data

!\[Extract Demo](screenshots/extract-demo.png)



\### Compare Documents

!\[Compare Demo](screenshots/compare-demo.png)

## Setup in 5 Steps

### Step 1 — Get Free Groq API Key

→ https://console.groq.com/keys → Sign up → Create API Key
(Starts with `gsk\\\_...`)

### Step 2 — Install dependencies

```bash
cd backend
python -m venv venv
venv\\\\Scripts\\\\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Step 3 — Set API Key

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux
```

Open `.env` and paste your Groq key:

```
GROQ\\\_API\\\_KEY=gsk\\\_your\\\_key\\\_here
```

### Step 4 — Start backend

```bash
python server.py
```

You should see: `✅ DocAI engine ready`

### Step 5 — Open frontend

Double-click `frontend/index.html` OR run:

```bash
cd frontend \\\&\\\& python -m http.server 3000
```

Then open `http://localhost:3000`

\---

## Resume Bullet Points

```
• Built AI Document Intelligence Platform using LangChain RAG + ChromaDB + Groq Llama 3 —
  multi-document Q\\\&A with source citations, structured data extraction from PDFs/DOCX,
  and document comparison. Deployed as REST API with FastAPI.
```

