"""
clean_excel.py
--------------
Run once to clean MTDR Excel and index into ChromaDB
Usage: python clean_excel.py --file "MTDR Records.xlsx"
"""

import argparse
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Config
COLLECTION_NAME = "mtdr_records"
DB_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

COL_MACHINE  = ["machine", "machine name", "equipment", "asset"]
COL_PROBLEM  = ["problem", "issue", "fault", "description", "breakdown"]
COL_SOLUTION = ["solution", "action taken", "fix", "resolution", "remedy"]
COL_LINE     = ["line", "line no", "production line", "line name"]

def find_col(df_cols, aliases):
    lower_cols = {c.lower().strip(): c for c in df_cols}
    for alias in aliases:
        if alias in lower_cols:
            return lower_cols[alias]
    return None

def clean_text(val):
    if pd.isna(val):
        return ""
    text = str(val).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def load_and_clean(filepath):
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = df.columns.str.strip()

    machine_col  = find_col(df.columns, COL_MACHINE)
    problem_col  = find_col(df.columns, COL_PROBLEM)
    solution_col = find_col(df.columns, COL_SOLUTION)
    line_col     = find_col(df.columns, COL_LINE)

    if not all([machine_col, problem_col, solution_col]):
        raise ValueError("Missing required columns")

    cleaned = pd.DataFrame({
        "machine": df[machine_col].apply(clean_text),
        "line": df[line_col].apply(clean_text) if line_col else "",
        "problem": df[problem_col].apply(clean_text),
        "solution": df[solution_col].apply(clean_text),
    })

    cleaned = cleaned[
        (cleaned["problem"].str.len() > 5) &
        (cleaned["solution"].str.len() > 5)
    ].drop_duplicates()

    return cleaned.reset_index(drop=True)

def index_to_chromadb(df):
    client = chromadb.PersistentClient(path=DB_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    docs, metas, ids = [], [], []

    for i, row in df.iterrows():
        docs.append(f"Machine: {row['machine']} Line: {row['line']} Problem: {row['problem']}")
        metas.append({
            "machine": row["machine"],
            "line": row["line"],
            "problem": row["problem"],
            "solution": row["solution"]
        })
        ids.append(f"id_{i}")

    collection.add(documents=docs, metadatas=metas, ids=ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    df = load_and_clean(args.file)
    index_to_chromadb(df)

    print("✅ Data indexed successfully")

if __name__ == "__main__":
    main()
