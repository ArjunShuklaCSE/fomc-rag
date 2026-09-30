"""HTTP API and UI.

    uv run uvicorn fomcrag.api:app --port 8000

GET  /            single-page UI
GET  /health      status, serving config, LLM backend
GET  /search      ?q=...&k=10  ranked chunks with scores and page links
POST /answer      {"question": "..."}  grounded answer with citations (503 if no LLM is configured)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import answer as qa
from .llm import backend_name, provider

STATIC = Path(__file__).parent / "static"
MAX_Q = 500
app = FastAPI(title="fomc-rag", version="1.0")


def llm_available() -> bool:
    if os.environ.get("FOMC_RETRIEVAL_ONLY") == "1":
        return False
    if provider() == "anthropic":
        return True
    import torch
    return torch.cuda.is_available()


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_Q)


@app.on_event("startup")
def warm() -> None:
    qa.sources("warm-up query about the federal funds rate")  # load index, embedder and reranker at startup


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    retriever, _ = qa.serving()
    return {"status": "ok", "config": retriever.cfg, "chunks": len(retriever.index.chunks),
            "llm": backend_name() if llm_available() else None, "abstain_threshold": qa.threshold()}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=MAX_Q), k: int = Query(10, ge=1, le=20)) -> dict:
    srcs, res = qa.sources(q, k=k)
    return {"query": q, "results": srcs, "timing_ms": res["timing"], "filtered": res["filtered"]}


@app.post("/answer")
def answer(req: AnswerRequest) -> dict:
    if not llm_available():
        raise HTTPException(503, "No LLM configured on this deployment: set ANTHROPIC_API_KEY, or use /search.")
    return qa.answer(req.question)
