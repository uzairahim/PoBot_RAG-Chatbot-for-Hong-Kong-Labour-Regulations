"""Document extraction, cleaning, and chunking.

Turns the raw PDFs/HTML in data/raw/ into clean, overlapping text chunks with
provenance metadata (source id, title, publisher, url, page/section). These
chunks are what we embed and later cite.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

# Chunks shorter than this (chars) are dropped as noise (stray headings,
# table fragments, page artifacts) — too little context to retrieve on.
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    """A single retrievable unit of text plus where it came from."""
    chunk_id: str
    source_id: str
    title: str
    publisher: str
    url: str
    category: str
    chunk_index: int
    text: str
    # Optional locator (PDF page number) to make citations more precise.
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(path: Path) -> list[str]:
    """Return the raw text of each PDF page (index 0 == page 1)."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


# Boilerplate lines commonly injected by gov.hk / labour.gov.hk page templates.
_HTML_BOILERPLATE = re.compile(
    r"skip to (main )?content|outdated browser|text size|"
    r"you are here|back to top|last (revision|review) date|"
    r"print this page|share (this|to)|font size|traditional chinese|"
    r"simplified chinese|copyright|all rights reserved",
    re.IGNORECASE,
)


def extract_html_text(path: Path) -> str:
    """Extract readable body text from an HTML file, dropping nav/chrome."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")

    # Drop non-content elements outright.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "form", "aside", "button", "svg", "iframe"]):
        tag.decompose()

    # Prefer a main-content container when the template exposes one. Gov.hk
    # templates include empty skip-link anchors like <a id="main-content">, so
    # pick the candidate with the MOST text rather than the first id match.
    candidates = [soup.find("main"), soup.find("article")]
    candidates += soup.find_all(id=re.compile(r"content|main", re.I))
    candidates += soup.find_all(class_=re.compile(r"content|main-|article", re.I))
    candidates.append(soup.body)
    candidates = [c for c in candidates if c is not None]

    main = max(candidates, key=lambda c: len(c.get_text(strip=True))) if candidates else soup
    text = main.get_text(separator="\n")

    # Line-level cleanup: drop boilerplate and empty lines.
    kept = [ln.strip() for ln in text.splitlines()
            if ln.strip() and not _HTML_BOILERPLATE.search(ln)]
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers that repeat across most PDF pages."""
    if len(pages) < 3:
        return pages
    line_counts: Counter[str] = Counter()
    for pg in pages:
        for ln in {ln.strip() for ln in pg.splitlines() if ln.strip()}:
            line_counts[ln] += 1
    threshold = max(3, int(len(pages) * 0.5))
    repeated = {ln for ln, c in line_counts.items()
                if c >= threshold and len(ln) < 120}
    cleaned = []
    for pg in pages:
        cleaned.append("\n".join(ln for ln in pg.splitlines()
                                 if ln.strip() not in repeated))
    return cleaned


_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize whitespace, de-hyphenate line breaks, drop page-number lines."""
    # Join words split across a line break by a hyphen: "employ-\nment" -> "employment".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or _PAGE_NUM.match(ln):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    # Collapse runs of blank lines and horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Sentence boundary: end punctuation followed by whitespace. Kept deliberately
# simple and robust for legal/administrative prose.
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into ~chunk_size-char chunks with char overlap.

    Overlap is realized by carrying the trailing sentences of the previous chunk
    (up to `overlap` chars) into the next one, preserving context across splits.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        # A single over-long sentence becomes its own chunk (hard split).
        if len(sent) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, len(sent), chunk_size):
                chunks.append(sent[i:i + chunk_size])
            continue

        if current_len + len(sent) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Build overlap tail from the end of the just-finished chunk.
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail[:], tail_len

        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Orchestration: raw file -> list[Chunk]
# ---------------------------------------------------------------------------

def cleaned_full_text(source: dict, raw_dir: Path) -> str:
    """Full cleaned text of a source, as a single readable string.

    NOTE: This is a *diagnostic / deliverable* helper for humans to inspect the
    cleaning quality — the retrieval pipeline does NOT consume its output (it
    uses the chunks from `process_source`). Writing these files to disk is gated
    behind `settings.write_processed_text` in src/ingest.py.
    """
    path = raw_dir / source["filename"]
    if source["type"] == "pdf":
        pages = _strip_repeated_lines(extract_pdf_pages(path))
        parts = []
        for page_no, page_text in enumerate(pages, start=1):
            cleaned = clean_text(page_text)
            if cleaned:
                parts.append(f"[page {page_no}]\n{cleaned}")
        return "\n\n".join(parts)
    return clean_text(extract_html_text(path))


def process_source(source: dict, raw_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Extract, clean, and chunk one source described by a sources.json entry."""
    path = raw_dir / source["filename"]
    chunks: list[Chunk] = []

    def _make(idx: int, body: str, page: int | None) -> Chunk:
        return Chunk(
            chunk_id=f"{source['id']}::{idx}",
            source_id=source["id"],
            title=source["title"],
            publisher=source["publisher"],
            url=source["url"],
            category=source["category"],
            chunk_index=idx,
            text=body,
            page=page,
        )

    if source["type"] == "pdf":
        pages = _strip_repeated_lines(extract_pdf_pages(path))
        idx = 0
        for page_no, page_text in enumerate(pages, start=1):
            cleaned = clean_text(page_text)
            if not cleaned:
                continue
            for body in chunk_text(cleaned, chunk_size, overlap):
                if len(body) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(_make(idx, body, page_no))
                idx += 1
    else:  # html
        cleaned = clean_text(extract_html_text(path))
        idx = 0
        for body in chunk_text(cleaned, chunk_size, overlap):
            if len(body) < MIN_CHUNK_CHARS:
                continue
            chunks.append(_make(idx, body, None))
            idx += 1

    return chunks
