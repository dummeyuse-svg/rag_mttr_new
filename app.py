"""
app.py
------
FastAPI backend for the MTDR local AI assistant.
Run: uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

import json
import re
from pathlib import Path
from typing import Optional

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "mtdr_records"
DB_PATH          = "./chroma_db"
EMBED_MODEL      = "all-MiniLM-L6-v2"
OLLAMA_URL       = "http://127.0.0.1:11434"
OLLAMA_MODEL     = "phi3"          # change to "llama3" if you pulled that
TOP_K            = 4                # how many past records to retrieve
MAX_TOKENS       = 200

# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTDR Local AI Assistant")

_client = chromadb.PersistentClient(path=DB_PATH)
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

def get_collection():
    try:
        return _client.get_collection(name=COLLECTION_NAME, embedding_function=_ef)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="MTDR database not found. Run clean_excel.py first."
        )


# ── Request / Response models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None   # optional: limit to a specific machine


class RecordMatch(BaseModel):
    machine: str
    problem: str
    solution: str
    similarity: float


class QueryResponse(BaseModel):
    ai_suggestion: str
    matched_records: list[RecordMatch]


# ── Helper: call Ollama locally ──────────────────────────────────────────────
async def ask_ollama(prompt: str) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": 0.2},
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve"
            )


# ── Main query endpoint ───────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_records(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    collection = get_collection()

    # Build ChromaDB where filter if machine is specified
    where = None
    if req.machine_filter and req.machine_filter.strip():
        where = {"machine": {"$eq": req.machine_filter.strip()}}

    # Retrieve top-K similar past records
    results = collection.query(
        query_texts=[req.query],
        n_results=min(TOP_K, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    matched: list[RecordMatch] = []
    context_blocks = []

    for meta, dist in zip(metadatas, distances):
        similarity = round(1 - dist, 3)   # cosine: distance 0 = perfect match
        matched.append(RecordMatch(
            machine=meta.get("machine", "Unknown"),
            problem=meta.get("problem", ""),
            solution=meta.get("solution", ""),
            similarity=similarity,
        ))
        context_blocks.append(
            f"Machine: {meta['machine']}\n"
            f"Problem: {meta['problem']}\n"
            f"Solution: {meta['solution']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a maintenance expert assistant for an industrial facility.
A technician is facing the following problem:

PROBLEM: {req.query}

Below are the most relevant past MTDR (Machine Trouble & Daily Report) records from the database:

{context}

Based on these past records, provide a clear, step-by-step recommended solution for the technician.
- Be concise and practical.
- Reference which past record is most applicable if relevant.
- If no record is a good match, say so and suggest basic troubleshooting steps.
- Do NOT make up solutions that aren't supported by the records.
"""

    ai_suggestion = await ask_ollama(prompt)

    return QueryResponse(ai_suggestion=ai_suggestion, matched_records=matched)


# ── List all unique machines ──────────────────────────────────────────────────
@app.get("/machines")
async def list_machines():
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    machines = sorted(set(m.get("machine", "") for m in results["metadatas"] if m.get("machine")))
    return {"machines": machines}


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        col = get_collection()
        count = col.count()
    except Exception:
        count = 0
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"records_indexed": count, "ollama_running": ollama_ok}


# ── Serve the UI (index.html must be in same folder) ─────────────────────────
@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>index.html not found. Place it in the same folder as app.py</h1>")
    return FileResponse(ui_path)
