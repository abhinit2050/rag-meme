"""
Step 3 of the Meme History RAG pipeline: embed each chunk (data/chunks/*.json)
and upsert it into MongoDB Atlas Vector Search.

Database: meme_history_rag
Collection: meme_chunks
Vector index: meme_chunks_vector_index (created here if missing)

Each stored document's _id is "<meme_slug>::<section>", so re-running this
script overwrites existing chunks instead of duplicating them.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"

DB_NAME = "meme_history_rag"
COLLECTION_NAME = "meme_chunks"
INDEX_NAME = "meme_chunks_vector_index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def ensure_vector_index(collection) -> None:
    existing = list(collection.list_search_indexes(INDEX_NAME))
    if existing:
        print(f"[index] '{INDEX_NAME}' already exists")
        return

    print(f"[index] creating '{INDEX_NAME}' ...")
    model = SearchIndexModel(
        name=INDEX_NAME,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIMENSIONS,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "meme_slug"},
            ]
        },
    )
    collection.create_search_index(model=model)

    while True:
        status = list(collection.list_search_indexes(INDEX_NAME))
        if status and status[0].get("queryable"):
            print(f"[index] '{INDEX_NAME}' is queryable")
            break
        print("[index] waiting for Atlas to finish building the index...")
        time.sleep(5)


def main() -> None:
    mongodb_uri = os.environ["MONGODB_URI"]
    chunks = load_chunks()
    print(f"[load] {len(chunks)} chunks from {CHUNKS_DIR}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"[embed] requesting {len(texts)} embeddings from OpenAI ({EMBEDDING_MODEL})...")
    vectors = embeddings.embed_documents(texts)

    client = MongoClient(mongodb_uri)
    collection = client[DB_NAME][COLLECTION_NAME]

    for chunk, vector in zip(chunks, vectors):
        doc_id = f"{chunk['meme_slug']}::{chunk['section']}"
        doc = {**chunk, "embedding": vector}
        collection.replace_one({"_id": doc_id}, doc, upsert=True)

    print(f"[stored] {len(chunks)} chunks upserted into {DB_NAME}.{COLLECTION_NAME}")

    ensure_vector_index(collection)


if __name__ == "__main__":
    main()
