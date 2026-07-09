"""Download the official source documents listed in data/sources.json.

Reproducible data-collection step. Run once before ingestion:

    python -m src.fetch_sources

Already-downloaded files are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import sys

import requests

from src.config import RAW_DIR, SOURCES_FILE

# Some gov.hk endpoints reject the default python-requests User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def load_sources() -> list[dict]:
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["sources"]


def download(source: dict, force: bool = False) -> bool:
    dest = RAW_DIR / source["filename"]
    if dest.exists() and not force:
        print(f"  [skip] {source['filename']} already exists ({dest.stat().st_size:,} bytes)")
        return True
    try:
        resp = requests.get(source["url"], headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [FAIL] {source['id']}: {exc}", file=sys.stderr)
        return False
    dest.write_bytes(resp.content)
    print(f"  [ok]   {source['filename']:<45} {len(resp.content):>10,} bytes  <- {source['url']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HK labour regulation source documents.")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    print(f"Fetching {len(sources)} sources into {RAW_DIR} ...")
    ok = sum(download(s, force=args.force) for s in sources)
    print(f"\nDone: {ok}/{len(sources)} sources available.")
    return 0 if ok == len(sources) else 1


if __name__ == "__main__":
    raise SystemExit(main())
