# meme-history-rag

A conversational RAG (retrieval-augmented generation) agent for internet meme history. Answers are grounded only in retrieved source material from [Know Your Meme](https://knowyourmeme.com) — it cites the meme name and source URL for every answer, and declines rather than guessing when a question falls outside the ingested corpus.

## Stack (planned)

- **Python / FastAPI** — HTTP API
- **LangChain** — chunking, retrieval, and the conversational chain
- **MongoDB Atlas Vector Search** — embedding storage and similarity search
- **Know Your Meme** — source corpus (~50-100 hand-picked "confirmed" entries to start, scaling to ~500)

## Status

Early stage — only the data collection step exists so far.

### Step 1: fetch raw entries (done)

`scripts/fetch_kym_entries.py` reads meme slugs from `data/meme_list.txt`, fetches each `https://knowyourmeme.com/memes/<slug>` page, and saves the extracted JSON-LD metadata, infobox (status/type/year/origin/tags), and body sections (about/origin/spread/various-examples/external-references) to `data/raw/<slug>.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/fetch_kym_entries.py
```

Requests are rate-limited (2.5s apart) and only hit `/memes/<slug>` pages, which `knowyourmeme.com/robots.txt` permits. Already-fetched slugs are skipped on re-run.

### Remaining steps (not yet built)

1. Chunking — split each fetched entry into chunks (origin/spread/variations), tagged with meme name, source URL, date scraped
2. Embedding + storage — embed chunks into MongoDB Atlas Vector Search
3. Retrieval — embed a user query, top-k similarity search
4. Grounded generation — LLM answers only from retrieved chunks, always citing meme name/source URL
5. Conversational wrapper — multi-turn chat history via LangChain
6. Testing — confirm out-of-corpus questions get a graceful decline, not a hallucinated answer

### Deferred to later phases

- Periodic re-scraping/re-ingestion (Know Your Meme entries get revised and new ones added)
- Image input: upload a meme image → CLIP embedding → match against stored meme image embeddings → retrieve that meme's history

## License

MIT — see [LICENSE](LICENSE).
