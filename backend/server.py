"""
DocAI - FastAPI Server
Professional REST API for document intelligence
"""

import os
import json
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from document_engine import DocumentEngine, DocumentInfo


# ─────────────────────────────────────────
#  App Lifecycle
# ─────────────────────────────────────────

_engine: Optional[DocumentEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    groq_key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if groq_key:
        try:
            _engine = DocumentEngine(groq_api_key=groq_key, model=model)
            print("✅ DocAI engine ready")
        except Exception as e:
            print(f"⚠️ Engine init failed: {e}")
    yield


app = FastAPI(
    title="DocAI — Document Intelligence API",
    description="Professional AI-powered document analysis platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────

class InitRequest(BaseModel):
    groq_api_key: str
    model: str = "llama-3.3-70b-versatile"

class QuestionRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None

class ExtractRequest(BaseModel):
    doc_id: str
    extraction_type: str = "general"  # invoice | contract | resume | general

class CompareRequest(BaseModel):
    doc_id_a: str
    doc_id_b: str


def require_engine():
    if _engine is None:
        raise HTTPException(503, "Engine not initialized. POST /init with your GROQ_API_KEY first.")
    return _engine


# ─────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": _engine is not None,
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "doc_count": len(_engine.documents) if _engine else 0,
    }


@app.post("/init")
async def initialize(req: InitRequest):
    global _engine
    try:
        _engine = DocumentEngine(groq_api_key=req.groq_api_key, model=req.model)
        os.environ["GROQ_API_KEY"] = req.groq_api_key
        os.environ["GROQ_MODEL"] = req.model
        return {"success": True, "message": "DocAI engine initialized ✅"}
    except Exception as e:
        raise HTTPException(500, f"Init failed: {str(e)}")


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document (PDF, DOCX, TXT)."""
    engine = require_engine()

    allowed = {".pdf", ".docx", ".doc", ".txt", ".md"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported. Use: {', '.join(allowed)}")

    max_size = 20 * 1024 * 1024  # 20MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(400, "File too large. Max 20MB.")

    try:
        info = engine.ingest_document(content, file.filename)
        return {
            "success": True,
            "document": {
                "doc_id": info.doc_id,
                "filename": info.filename,
                "file_type": info.file_type,
                "page_count": info.page_count,
                "word_count": info.word_count,
                "chunk_count": info.chunk_count,
                "summary": info.summary,
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    engine = require_engine()
    docs = engine.list_documents()
    return {
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "file_type": d.file_type,
                "page_count": d.page_count,
                "word_count": d.word_count,
                "chunk_count": d.chunk_count,
                "summary": d.summary,
            }
            for d in docs
        ]
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    engine = require_engine()
    engine.remove_document(doc_id)
    return {"success": True, "message": f"Document {doc_id} removed"}


@app.post("/qa")
async def question_answer(req: QuestionRequest):
    """Ask a question across one or more documents."""
    engine = require_engine()
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        result = engine.answer_question(req.question, req.doc_ids)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "doc_ids_used": result.doc_ids_used,
        }
    except Exception as e:
        raise HTTPException(500, f"Q&A failed: {str(e)}")


@app.post("/extract")
async def extract_data(req: ExtractRequest):
    """Extract structured data from a document."""
    engine = require_engine()
    try:
        result = engine.extract_structured(req.doc_id, req.extraction_type)
        return {
            "extracted_data": result.extracted_data,
            "doc_id": result.doc_id,
        }
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {str(e)}")


@app.post("/compare")
async def compare_docs(req: CompareRequest):
    """Compare two documents."""
    engine = require_engine()
    if req.doc_id_a == req.doc_id_b:
        raise HTTPException(400, "Please select two different documents to compare.")
    try:
        result = engine.compare_documents(req.doc_id_a, req.doc_id_b)
        return {
            "similarities": result.similarities,
            "differences": result.differences,
            "summary": result.summary,
            "doc_a_name": result.doc_a_name,
            "doc_b_name": result.doc_b_name,
        }
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")


@app.get("/models")
async def list_models():
    return {"models": [
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Best Quality)"},
        {"id": "llama-3.1-8b-instant",   "name": "Llama 3.1 8B (Fastest)"},
        {"id": "mixtral-8x7b-32768",      "name": "Mixtral 8x7B (32k context)"},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
