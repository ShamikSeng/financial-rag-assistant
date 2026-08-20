"""
Freezes the corpus: downloads the PDF for every paper in curated_candidates.json,
computes sha256 for each, and writes data/papers/corpus_manifest.json as the
permanent provenance record (source URL, hash, download date) for what's in the
corpus and where it came from.

This is the point of no return per PROJECT_LOG.md's eval-staging rule: once this
runs and Phase 1's baseline is measured against it, the corpus shouldn't be added
to or trimmed without re-running that baseline -- otherwise later pipeline-variant
comparisons (hybrid, rerank, ...) stop being apples-to-apples.

Usage:
    python scripts/corpus_intake/freeze_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "papers"
RAW_DIR = DATA_DIR / "raw"
ARXIV_REQUEST_DELAY_S = 3.0  # arXiv API TOS asks for >=3s between requests


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pdf(pdf_url: str, dest: Path) -> None:
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> None:
    curated = json.loads((DATA_DIR / "curated_candidates.json").read_text(encoding="utf-8"))
    papers = curated["kept"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for i, p in enumerate(papers):
        aid = p["arxiv_id"]
        dest = RAW_DIR / f"{aid}.pdf"
        print(f"[{i + 1}/{len(papers)}] {aid} - {p['title'][:60]}", file=sys.stderr)

        if dest.exists() and dest.stat().st_size > 0:
            print("  already downloaded, skipping fetch", file=sys.stderr)
        else:
            # Use the unversioned URL, not the feed's raw pdf_url -- the arXiv
            # Atom feed sometimes points at a specific version (e.g. ...v2) that
            # a later revision has since superseded, which 404s. The unversioned
            # URL always redirects to whatever the current version is.
            unversioned_url = f"https://arxiv.org/pdf/{aid}"
            try:
                download_pdf(unversioned_url, dest)
            except requests.exceptions.RequestException as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                manifest_entries.append({
                    "arxiv_id": aid,
                    "title": p["title"],
                    "status": "download_failed",
                    "error": str(e),
                })
                time.sleep(ARXIV_REQUEST_DELAY_S)
                continue
            time.sleep(ARXIV_REQUEST_DELAY_S)

        manifest_entries.append({
            "arxiv_id": aid,
            "title": p["title"],
            "authors": p.get("authors", []),
            "primary_category": p.get("primary_category"),
            "published": p.get("published"),
            "source": p["source"],  # "anchor" or "arxiv_search"
            "pdf_url": p["pdf_url"],
            "local_path": str(dest.relative_to(DATA_DIR.parents[1])).replace("\\", "/"),
            "sha256": sha256_of(dest),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": dest.stat().st_size,
            "semantic_scholar_found": p.get("s2_found"),
            "citation_count": p.get("citation_count"),
            "status": "ok",
        })

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "total_papers": len(manifest_entries),
        "papers": manifest_entries,
    }
    manifest_path = DATA_DIR / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = sum(1 for e in manifest_entries if e["status"] == "ok")
    failed = len(manifest_entries) - ok
    total_bytes = sum(e.get("file_size_bytes", 0) for e in manifest_entries if e["status"] == "ok")
    print(f"\n{ok} downloaded OK, {failed} failed, {total_bytes / 1e6:.1f} MB total", file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
