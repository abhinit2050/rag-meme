"""
Step 1 of the Meme History RAG pipeline: fetch raw entry data from
Know Your Meme (knowyourmeme.com) and save it locally as JSON.

For each slug listed in data/meme_list.txt, downloads
https://knowyourmeme.com/memes/<slug> and extracts:
  - JSON-LD metadata (headline, description, dates, keywords)
  - the infobox (status, type, year, origin site, tags)
  - body sections (about, origin, spread, various-examples, external-references)

Only fetches /memes/<slug> pages, which knowyourmeme.com/robots.txt permits
(it disallows /comments, /search, /users, /forums, etc., none of which this
script touches). Requests are rate-limited and already-fetched slugs are
skipped on re-run. No chunking or embedding happens here -- that's step 2.
"""

import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://knowyourmeme.com/memes/{slug}"
USER_AGENT = "MemeHistoryRAGBot/0.1 (personal research project; contact: abhinit1990@gmail.com)"
REQUEST_DELAY_SECONDS = 2.5
REQUEST_TIMEOUT_SECONDS = 15

ROOT = Path(__file__).resolve().parent.parent
MEME_LIST_PATH = ROOT / "data" / "meme_list.txt"
RAW_DIR = ROOT / "data" / "raw"

SECTION_IDS = [
    "about",
    "origin",
    "spread",
    "various-examples",
    "notable-examples",
    "external-references",
]


def load_slugs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def fetch_html(slug: str) -> str:
    url = BASE_URL.format(slug=slug)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text


def extract_json_ld(soup: BeautifulSoup) -> dict:
    tag = soup.find("script", type="application/ld+json")
    if not tag or not tag.string:
        return {}
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return {}
    return {
        "headline": data.get("headline"),
        "description": data.get("description"),
        "date_published": data.get("datePublished"),
        "date_modified": data.get("dateModified"),
        "keywords": data.get("keywords"),
    }


def extract_infobox(soup: BeautifulSoup) -> dict:
    infobox: dict = {}
    entry_body = soup.find("div", id="entry_body")
    if not entry_body:
        return infobox

    dl = entry_body.find("dl")
    if dl:
        for dt in dl.find_all("dt"):
            key = dt.get_text(strip=True).rstrip(":").lower()
            dd = dt.find_next_sibling("dd")
            if dd:
                infobox[key] = dd.get_text(" ", strip=True)

    tags_dl = entry_body.find("dl", id="entry_tags")
    if tags_dl:
        infobox["tags"] = [a.get_text(strip=True) for a in tags_dl.find_all("a")]

    return infobox


def extract_sections(soup: BeautifulSoup) -> dict:
    sections: dict = {}
    for section_id in SECTION_IDS:
        header = soup.find("h2", id=section_id)
        if not header:
            continue
        parts = []
        for sibling in header.find_next_siblings():
            if sibling.name == "h2":
                break
            text = sibling.get_text(" ", strip=True)
            if text:
                parts.append(text)
        text = " ".join(parts).strip()
        if text:
            sections[section_id] = text
    return sections


def fetch_meme(slug: str) -> dict:
    html = fetch_html(slug)
    soup = BeautifulSoup(html, "html.parser")
    return {
        "slug": slug,
        "source_url": BASE_URL.format(slug=slug),
        **extract_json_ld(soup),
        "infobox": extract_infobox(soup),
        "sections": extract_sections(soup),
    }


def main() -> None:
    slugs = load_slugs(MEME_LIST_PATH)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for i, slug in enumerate(slugs):
        out_path = RAW_DIR / f"{slug}.json"
        if out_path.exists():
            print(f"[skip] {slug} already fetched")
            continue

        print(f"[fetch] {slug}")
        try:
            data = fetch_meme(slug)
        except requests.HTTPError as e:
            print(f"[error] {slug}: HTTP {e.response.status_code}", file=sys.stderr)
            continue
        except requests.RequestException as e:
            print(f"[error] {slug}: {e}", file=sys.stderr)
            continue

        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out_path}")

        if i < len(slugs) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()
