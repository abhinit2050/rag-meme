# meme-history-rag

A conversational RAG (retrieval-augmented generation) agent for internet meme history. Answers are grounded only in retrieved source material from [Know Your Meme](https://knowyourmeme.com) — it cites the meme name and source URL for every answer, and declines rather than guessing when a question falls outside the ingested corpus.

## Stack (planned)

- **Python / FastAPI** — HTTP API
- **LangChain** — chunking, retrieval, and the conversational chain
- **MongoDB Atlas Vector Search** — embedding storage and similarity search
- **Know Your Meme** — source corpus (~50-100 hand-picked "confirmed" entries to start, scaling to ~500)

## Status

Pipeline validated end-to-end (steps 1-6) against a seed set of 8 memes: rickroll, distracted-boyfriend, doge, pepe-the-frog, trollface, bad-luck-brian, success-kid-i-hate-sandcastles, overly-attached-girlfriend. Deliberately kept to 8 while proving out the pipeline before scaling the corpus toward 50-100.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root with:

```
MONGODB_URI=<your MongoDB Atlas connection string>
OPENAI_API_KEY=<your OpenAI API key>
```

### Step 1: fetch raw entries (done)

`scripts/fetch_kym_entries.py` reads meme slugs from `data/meme_list.txt`, fetches each `https://knowyourmeme.com/memes/<slug>` page, and saves the extracted JSON-LD metadata, infobox (status/type/year/origin/tags), and body sections (about/origin/spread/various-examples/external-references) to `data/raw/<slug>.json`.

```bash
python3 scripts/fetch_kym_entries.py
```

Requests are rate-limited (2.5s apart) and only hit `/memes/<slug>` pages, which `knowyourmeme.com/robots.txt` permits. Already-fetched slugs are skipped on re-run.

### Step 2: chunking + metadata (done)

`scripts/chunk_entries.py` splits each `data/raw/<slug>.json` into one chunk per body section (about/origin/spread/various-examples/external-references), tagged with `meme_slug`, `meme_name`, `section`, `source_url`, and `date_scraped`. Output goes to `data/chunks/<slug>.json`.

```bash
python3 scripts/chunk_entries.py
```

Sections are used as chunk boundaries as-is (no further length-based splitting yet) — worth revisiting if a very long section (e.g. Rickroll's `spread`, ~5,700 words spanning 15 years of history) turns out to hurt retrieval precision.

### Step 3: embedding + storage (done)

`scripts/embed_and_store.py` embeds every chunk with OpenAI's `text-embedding-3-small` and upserts it into MongoDB Atlas (`meme_history_rag.meme_chunks`), creating the `meme_chunks_vector_index` Atlas Vector Search index if it doesn't already exist. Each document's `_id` is `<meme_slug>::<section>`, so re-running is idempotent.

```bash
python3 scripts/embed_and_store.py
```

Requires your Atlas cluster's Network Access list to include your current IP, or the connection fails with a TLS handshake error rather than a clear authentication error.

### Step 4: retrieval (done)

`scripts/retrieve.py` embeds a question and runs a `$vectorSearch` aggregation against `meme_chunks`, returning the top-k chunks with similarity scores and source metadata.

```bash
python3 scripts/retrieve.py "Why is Rickroll called Rickroll?"
```

### Step 5: grounded generation (done)

`scripts/generate.py` retrieves the top-k chunks for a question, drops any chunk scoring below `MIN_RELEVANCE_SCORE` (0.72 — chosen from observing ~0.78-0.82 for genuinely relevant chunks vs. ~0.67 for unrelated ones), and only then asks `gpt-4o-mini` to answer using solely those chunks, citing meme name + source URL. If nothing clears the relevance bar, the LLM is never called and a fixed decline message is returned instead — this two-layer guard (score filter + prompt instruction) is what keeps it from hallucinating on out-of-corpus questions.

```bash
python3 scripts/generate.py "Who is the dog in the Doge meme?"
```

Verified manually: in-corpus questions get cited, grounded answers; out-of-corpus questions (e.g. "Tell me about the Grumpy Cat meme") get the decline message instead of the model falling back on its own training knowledge.

### Step 6: conversational wrapper (done)

`scripts/chat.py` is a REPL that adds multi-turn chat on top of step 5. Since vector search has no notion of prior turns, each follow-up question is first rewritten into a standalone query using the conversation history, then retrieval and generation proceed exactly as in step 5 (same score gate and decline path) — chat history informs query condensation and conversational tone, never the grounding itself.

```bash
python3 scripts/chat.py
```

Verified manually: a follow-up like "Why did it become popular?" after asking about Doge correctly condenses to "Why did the Doge meme become popular?" and retrieves the right chunks; an out-of-corpus follow-up (e.g. "Tell me about the Grumpy Cat meme") still declines rather than answering from the model's own knowledge.

### Remaining steps

7. Testing — a proper pass of out-of-corpus questions to confirm the decline path holds up beyond ad hoc spot checks

### Deferred to later phases

- Periodic re-scraping/re-ingestion (Know Your Meme entries get revised and new ones added)
- Image input: upload a meme image → CLIP embedding → match against stored meme image embeddings → retrieve that meme's history

## License

MIT — see [LICENSE](LICENSE).
