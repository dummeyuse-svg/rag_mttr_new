"""
app.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from collections import Counter
import httpx

COLLECTION_NAME = "mtdr_records"
DB_PATH = "./chroma_db"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral"

app = FastAPI()

client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection():
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None
    line_filter: Optional[str] = None

@app.post("/query")
async def query(req: QueryRequest):

    col = get_collection()

    where = {}
    if req.machine_filter:
        where["machine"] = {"$eq": req.machine_filter}
    if req.line_filter:
        where["line"] = {"$eq": req.line_filter}

    if not where:
        where = None

    results = col.query(
        query_texts=[req.query],
        n_results=10,
        where=where,
        include=["metadatas"]
    )

    metas = results["metadatas"][0]

    # 🔥 Frequency logic
    counter = Counter()
    for m in metas:
        counter[m["solution"]] += 1

    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Prepare LLM input
    summary = "\n".join([
        f"{i+1}. {sol} (used {count} times)"
        for i, (sol, count) in enumerate(ranked)
    ])

    prompt = f"""
Problem: {req.query}

Top solutions based on past data:
{summary}

Give best solution step-by-step.
"""

    async with httpx.AsyncClient() as client_http:
        res = await client_http.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

    ai_response = res.json()["response"]

    return {
        "ai_suggestion": ai_response,
        "ranked_solutions": [
            {"solution": s, "count": c} for s, c in ranked
        ],
        "matched_records": metas
    }
