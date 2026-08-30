"""
Step 4 of the Meme History RAG pipeline: given a user question, embed it and
run a MongoDB Atlas Vector Search similarity query against the stored chunks.

This only retrieves -- no LLM generation happens here yet (that's step 5).
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

load_dotenv()

DB_NAME = "meme_history_rag"
COLLECTION_NAME = "meme_chunks"
INDEX_NAME = "meme_chunks_vector_index"
EMBEDDING_MODEL = "text-embedding-3-small"


def retrieve(query: str, k: int = 5) -> list[dict]:
    client = MongoClient(os.environ["MONGODB_URI"])
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_vector = embeddings.embed_query(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "meme_name": 1,
                "meme_slug": 1,
                "section": 1,
                "text": 1,
                "source_url": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def main() -> None:
    query = " ".join(sys.argv[1:]) or "what is the history behind the meme - Ain't Nobody Got Time for That"
    print(f"[query] {query}\n")

    for i, result in enumerate(retrieve(query), start=1):
        print(f"{i}. {result['meme_name']} / {result['section']}  (score={result['score']:.4f})")
        print(f"   {result['source_url']}")
        print(f"   {result['text'][:500]}...\n")


if __name__ == "__main__":
    main()
