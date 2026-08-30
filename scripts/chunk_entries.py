"""
Step 2 of the Meme History RAG pipeline: turn each fetched meme entry
(data/raw/<slug>.json) into a list of retrieval-ready chunks
(data/chunks/<slug>.json).

Chunking strategy: one chunk per body section (about, origin, spread,
various-examples/notable-examples, external-references). Sections are
already natural, semantically distinct units in the source, so no further
splitting is applied yet -- revisit if a section proves too long for good
embedding/retrieval quality once we test with the LLM.

Each chunk carries the metadata needed to cite the source in an answer:
meme_slug, meme_name, section, source_url, date_scraped.
"""

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_DIR = ROOT / "data" / "chunks"


def scraped_date(raw_path: Path) -> str:
    return date.fromtimestamp(raw_path.stat().st_mtime).isoformat()


def chunk_entry(raw_path: Path) -> list[dict]:
    entry = json.loads(raw_path.read_text(encoding="utf-8"))
    meme_name = entry.get("headline") or entry["slug"]
    source_url = entry["source_url"]
    date_scraped = scraped_date(raw_path)

    chunks = []
    for section, text in entry.get("sections", {}).items():
        if not text.strip():
            continue
        chunks.append(
            {
                "meme_slug": entry["slug"],
                "meme_name": meme_name,
                "section": section,
                "text": text,
                "source_url": source_url,
                "date_scraped": date_scraped,
            }
        )
    return chunks


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(RAW_DIR.glob("*.json")):
        chunks = chunk_entry(raw_path)
        out_path = CHUNKS_DIR / raw_path.name
        out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[chunked] {raw_path.name} -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()
