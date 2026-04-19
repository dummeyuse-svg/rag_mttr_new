from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
import httpx
from collections import Counter

app = FastAPI()

COLLECTION_NAME = "mtdr_records"
DB_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection():
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None
    line_number: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    collection = get_collection()

    where = {}

    if req.machine_filter:
        where["machine"] = {"$eq": req.machine_filter}

    if req.line_number:
        where["machine"] = {"$contains": req.line_number}

    results = collection.query(
        query_texts=[req.query],
        n_results=10,
        where=where if where else None,
        include=["metadatas", "distances"]
    )

    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # 🔥 Frequency logic
    solution_counts = Counter([m["solution"] for m in metas])

    enriched = []
    for m, d in zip(metas, dists):
        enriched.append({
            "machine": m["machine"],
            "problem": m["problem"],
            "solution": m["solution"],
            "similarity": round(1 - d, 3),
            "freq": solution_counts[m["solution"]]
        })

    # 🔥 Sort by frequency first, then similarity
    enriched.sort(key=lambda x: (-x["freq"], -x["similarity"]))

    context = "\n\n".join(
        f"{e['problem']} -> {e['solution']}" for e in enriched
    )

    prompt = f"""
ONLY use the following records.

Problem: {req.query}

Records:
{context}

Give step-by-step solution.
Prioritize most frequent solution.
Do NOT invent anything.
"""

    async with httpx.AsyncClient() as client_http:
        r = await client_http.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        ai = r.json()["response"]

    return {
        "ai_suggestion": ai,
        "matched_records": enriched
    }


@app.get("/machines")
def machines():
    col = get_collection()
    data = col.get(include=["metadatas"])
    machines = list(set(m["machine"] for m in data["metadatas"]))
    return {"machines": machines}
