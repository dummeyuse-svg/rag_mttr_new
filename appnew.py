import json
import re
from pathlib import Path
from typing import Optional

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mtdr_records"
DB_PATH = "./chroma_db"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma:2b"

TOP_K = 6                 # Slightly increased for better context
MAX_TOKENS = 280          # Slightly increased for better explanations

# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTTR Local AI Assistant")

_client = chromadb.PersistentClient(path=DB_PATH)

# Local embedding model path
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="./local_model"
)


def get_collection():
    try:
        return _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="MTDR database not found. Run clean_excel.py first."
        )


# ── Request / Response Models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None


class RecordMatch(BaseModel):
    machine: str
    problem: str
    solution: str
    similarity: float


class QueryResponse(BaseModel):
    ai_suggestion: str
    matched_records: list[RecordMatch]


# ── Ollama Helper ────────────────────────────────────────────────────────────
async def ask_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.3
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload
            )

            response.raise_for_status()

            return response.json().get("response", "").strip()

        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve"
            )


# ── Main Query Endpoint ──────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_records(req: QueryRequest):

    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    collection = get_collection()

    # Optional machine filter
    where = None

    if req.machine_filter and req.machine_filter.strip():
        where = {
            "machine": {
                "$eq": req.machine_filter.strip()
            }
        }

    # Retrieve similar MTDR records
    results = collection.query(
        query_texts=[req.query],
        n_results=min(TOP_K, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matched: list[RecordMatch] = []
    context_blocks = []

    for meta, dist in zip(metadatas, distances):

        similarity = round(1 - dist, 3)

        matched.append(
            RecordMatch(
                machine=meta.get("machine", "Unknown"),
                problem=meta.get("problem", ""),
                solution=meta.get("solution", ""),
                similarity=similarity,
            )
        )

        # Improved structured context formatting
        context_blocks.append(
            f"""
MTDR RECORD

Machine:
{meta.get('machine', '')}

Problem Observed:
{meta.get('problem', '')}

Action Taken:
{meta.get('solution', '')}
"""
        )

    context = "\n\n-----------------------------\n\n".join(context_blocks)

    prompt = f"""You are an expert industrial maintenance technician AI assistant.

TECHNICIAN'S ISSUE:
{req.query}

PAST MAINTENANCE RECORDS (may be vague or incomplete):
{context}

Instructions:
- Use the records above as hints/context only.
- If a record's solution is vague (e.g. "updated software", "replaced part"), expand it using your own technical knowledge.
- Only state information you are confident is technically correct. Do not guess.
- Be concise and practical — operators need fast, actionable answers.

Respond in this exact format:

MOST LIKELY CAUSE: [1 sentence]

FIX: [2-4 numbered steps, short and direct]

OTHER CAUSES: [1-2 bullet points, one line each]

SAFETY CHECK: [1 critical precaution, 1 sentence]"""

    ai_suggestion = await ask_ollama(prompt)

    return QueryResponse(
        ai_suggestion=ai_suggestion,
        matched_records=matched
    )


# ── List All Unique Machines ─────────────────────────────────────────────────
@app.get("/machines")
async def list_machines():

    collection = get_collection()

    results = collection.get(include=["metadatas"])

    machines = sorted(
        set(
            m.get("machine", "")
            for m in results["metadatas"]
            if m.get("machine")
        )
    )

    return {"machines": machines}


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():

    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = response.status_code == 200

    except Exception:
        ollama_ok = False

    return {
        "records_indexed": count,
        "ollama_running": ollama_ok
    }


# ── Serve Frontend ───────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def serve_ui():

    ui_path = Path(__file__).parent / "index.html"

    if not ui_path.exists():
        return HTMLResponse(
            "<h1>index.html not found. Place it in the same folder as app.py</h1>"
        )

    return FileResponse(ui_path)
