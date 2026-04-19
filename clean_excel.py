import argparse
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "mtdr_records"
DB_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

COL_MACHINE  = ["machine", "machine name", "equipment", "asset"]
COL_PROBLEM  = ["problem", "issue", "fault", "description", "breakdown"]
COL_SOLUTION = ["solution", "action taken", "fix", "resolution", "remedy"]
COL_CATEGORY = ["category", "type", "issue type"]


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
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F]", "", text)
    return text.strip()


def load_and_clean(filepath):
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = df.columns.str.strip()

    machine_col  = find_col(df.columns, COL_MACHINE)
    problem_col  = find_col(df.columns, COL_PROBLEM)
    solution_col = find_col(df.columns, COL_SOLUTION)
    category_col = find_col(df.columns, COL_CATEGORY)

    cleaned = pd.DataFrame({
        "machine":  df[machine_col].apply(clean_text),
        "problem":  df[problem_col].apply(clean_text),
        "solution": df[solution_col].apply(clean_text),
        "category": df[category_col].apply(clean_text) if category_col else ""
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

    documents, metadatas, ids = [], [], []

    for i, row in df.iterrows():
        documents.append(f"Machine: {row['machine']}. Problem: {row['problem']}")
        metadatas.append({
            "machine": row["machine"],
            "problem": row["problem"],
            "solution": row["solution"],
            "category": row["category"]
        })
        ids.append(f"id_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(documents)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    df = load_and_clean(args.file)
    index_to_chromadb(df)
