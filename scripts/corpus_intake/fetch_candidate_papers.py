"""
Builds a CANDIDATE paper list for the RAG/retrieval-methods corpus. Nothing here
downloads PDFs or freezes anything -- it only produces a list for human review.

What it does:
  1. Queries the arXiv API for recent papers in cs.CL / cs.IR / cs.LG matching
     RAG/retrieval-adjacent keywords, within the last ~N years (default 3).
  2. Adds a curated set of older foundational papers (DPR, RAG, cross-encoder
     reranking, ColBERT, GraphRAG, etc.) as anchor nodes -- included regardless
     of the recency cutoff so the citation graph has real internal connectivity
     instead of every recent paper's references pointing outside the corpus.
  3. Looks up each candidate's references/citations via the Semantic Scholar
     Graph API and computes how many land on OTHER candidates in this same
     list (internal citation density) -- this is the signal for whether
     Phase 4 (GraphRAG) will actually have something to traverse.
  4. Writes:
       - data/papers/candidates.json  (full metadata, machine-readable)
       - data/papers/candidates.md    (human-readable review table)

Freezing the corpus (picking the final ~40-50, downloading PDFs, writing
corpus_manifest.json) is a deliberate follow-up step -- and per PROJECT_LOG.md's
eval-staging rule, it must happen BEFORE the Phase 1 eval baseline runs, not
be revisited mid-comparison.

Usage:
    python scripts/corpus_intake/fetch_candidate_papers.py
    python scripts/corpus_intake/fetch_candidate_papers.py --max-recent 60 --skip-s2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ARXIV_API = "http://export.arxiv.org/api/query"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

ARXIV_REQUEST_DELAY_S = 3.0   # arXiv API TOS asks for >=3s between requests
S2_REQUEST_DELAY_S = 1.5      # be polite to the unauthenticated S2 tier

CATEGORIES = ["cs.CL", "cs.IR", "cs.LG"]

KEYWORDS = [
    "retrieval-augmented generation",
    "hybrid retrieval",
    "dense retrieval",
    "cross-encoder reranking",
    "reranker",
    "GraphRAG",
    "knowledge graph retrieval",
    "long-context retrieval",
]

# Curated anchor papers: older/foundational (or otherwise off-window) papers,
# deliberately included so the citation graph has real internal edges instead
# of mostly pointing outside a recency-only corpus, AND so each project phase
# has at least one directly-relevant paper tied to it. Hand-curated, not
# fetched from a search -- every ID here has been spot-verified against the
# live arXiv API (title match) before being trusted; one candidate ID that
# failed that check was dropped rather than included on faith.
ANCHOR_PAPERS: list[tuple[str, str]] = [
    ("2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering"),
    ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("2002.08909", "REALM: Retrieval-Augmented Language Model Pre-Training"),
    ("1901.04085", "Passage Re-ranking with BERT"),
    ("2004.12832", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"),
    ("2404.16130", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"),
    ("2307.03172", "Lost in the Middle: How Language Models Use Long Contexts"),
    ("2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"),
    # Added after external review, each spot-verified against the live arXiv API:
    ("2112.01488", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"),
    ("2309.15217", "Ragas: Automated Evaluation of Retrieval Augmented Generation"),
    ("2401.15884", "Corrective Retrieval Augmented Generation"),
    ("2402.03216", "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"),
    ("2403.14403", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"),
    ("2407.01449", "ColPali: Efficient Document Retrieval with Vision Language Models"),
]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "papers"


# --- arXiv -------------------------------------------------------------------

def build_arxiv_query() -> str:
    cat_clause = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    kw_clause = " OR ".join(f'abs:"{k}"' for k in KEYWORDS)
    return f"({cat_clause}) AND ({kw_clause})"


def fetch_arxiv_page(query: str, start: int, max_results: int) -> str:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = entry.findtext(f"{ATOM_NS}id", default="")
        # raw_id looks like http://arxiv.org/abs/2401.01234v2
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        arxiv_id = arxiv_id.split("v")[0] if "v" in arxiv_id.rsplit("/", 1)[-1] else arxiv_id
        title = (entry.findtext(f"{ATOM_NS}title", default="") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{ATOM_NS}summary", default="") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{ATOM_NS}published", default="")
        authors = [
            a.findtext(f"{ATOM_NS}name", default="")
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        primary_cat_el = entry.find(f"{ARXIV_NS}primary_category")
        primary_cat = primary_cat_el.get("term") if primary_cat_el is not None else ""
        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors,
                "primary_category": primary_cat,
                "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                "source": "arxiv_search",
            }
        )
    return entries


def fetch_recent_candidates(max_recent: int, years_back: int) -> list[dict]:
    query = build_arxiv_query()
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years_back)

    collected: list[dict] = []
    seen_ids: set[str] = set()
    start = 0
    page_size = min(50, max_recent)

    while len(collected) < max_recent:
        print(f"  arXiv: fetching results {start}..{start + page_size}", file=sys.stderr)
        xml_text = fetch_arxiv_page(query, start=start, max_results=page_size)
        page = parse_arxiv_feed(xml_text)
        if not page:
            break

        stop = False
        for e in page:
            try:
                pub_dt = datetime.strptime(e["published"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pub_dt = None
            if pub_dt and pub_dt < cutoff:
                stop = True
                continue
            if e["arxiv_id"] in seen_ids:
                continue
            seen_ids.add(e["arxiv_id"])
            collected.append(e)
            if len(collected) >= max_recent:
                break

        start += page_size
        if stop or len(page) < page_size:
            break
        time.sleep(ARXIV_REQUEST_DELAY_S)

    return collected


def fetch_anchor_candidates() -> list[dict]:
    # arXiv's own per-ID lookup endpoint accepts a comma-joined id_list.
    ids = ",".join(aid for aid, _ in ANCHOR_PAPERS)
    params = {"id_list": ids, "max_results": len(ANCHOR_PAPERS)}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    entries = parse_arxiv_feed(resp.text)
    for e in entries:
        e["source"] = "anchor"
    return entries


# --- Semantic Scholar ----------------------------------------------------------

S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_CHUNK_SIZE = 50  # S2 allows up to 500/request; stay well under to keep payloads light


def fetch_s2_batch(arxiv_ids: list[str]) -> dict[str, dict | None]:
    """One POST per chunk instead of one GET per paper. This is the actual fix for
    the rate-limit wall the per-paper version hit: ~88 sequential unauthenticated
    GETs got 429'd almost across the board, whereas a couple of batch POSTs stay
    comfortably under the free tier's limits.
    """
    results: dict[str, dict | None] = {}
    # Only "references" (each paper's own bibliography) is needed to detect internal
    # edges -- it's small and bounded, unlike "citations" (who cites it), which is
    # unbounded and what made the very first per-paper run time out on a
    # highly-cited anchor. "A cites B" is fully captured by A's own references list.
    params = {"fields": "title,year,citationCount,references.externalIds"}

    for start in range(0, len(arxiv_ids), S2_BATCH_CHUNK_SIZE):
        chunk = arxiv_ids[start:start + S2_BATCH_CHUNK_SIZE]
        payload = {"ids": [f"ARXIV:{aid}" for aid in chunk]}
        print(f"  S2 batch: {start + 1}..{start + len(chunk)} of {len(arxiv_ids)}", file=sys.stderr)

        data = None
        for attempt in range(5):
            try:
                resp = requests.post(S2_BATCH_API, params=params, json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                wait = 5 * (attempt + 1)
                print(f"    S2 batch network error ({e.__class__.__name__}), retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    S2 batch rate-limited, backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break

        if data is None:
            print(f"    S2 batch giving up on this chunk after repeated failures "
                  f"({len(chunk)} papers marked not found)", file=sys.stderr)
            for aid in chunk:
                results[aid] = None
        else:
            # Response is a list aligned with the input order; null entries mean
            # S2 doesn't have that paper.
            for aid, entry in zip(chunk, data):
                results[aid] = entry

        time.sleep(S2_REQUEST_DELAY_S)

    return results


def extract_arxiv_ids(items: list[dict] | None) -> set[str]:
    out = set()
    for item in items or []:
        ext = (item or {}).get("externalIds") or {}
        aid = ext.get("ArXiv")
        if aid:
            out.add(aid)
    return out


def annotate_with_citations(candidates: list[dict]) -> None:
    arxiv_ids = [c["arxiv_id"] for c in candidates]
    batch_results = fetch_s2_batch(arxiv_ids)

    found = 0
    for c in candidates:
        data = batch_results.get(c["arxiv_id"])
        if data is None:
            c["s2_found"] = False
            c["ref_arxiv_ids"] = []
            c["citation_count"] = None
        else:
            found += 1
            c["s2_found"] = True
            c["ref_arxiv_ids"] = sorted(extract_arxiv_ids(data.get("references")))
            c["citation_count"] = data.get("citationCount")

    print(f"  S2: found {found}/{len(candidates)} papers", file=sys.stderr)


def compute_internal_edges(candidates: list[dict]) -> list[tuple[str, str]]:
    # "A cites B" is fully captured by A's own references list, so we only
    # need to walk references (not the reverse "citations" direction) -- see
    # the comment in fetch_s2_citation_data for why citations was dropped.
    corpus_ids = {c["arxiv_id"] for c in candidates}
    edges: set[tuple[str, str]] = set()
    for c in candidates:
        for ref_id in c.get("ref_arxiv_ids", []):
            if ref_id in corpus_ids and ref_id != c["arxiv_id"]:
                edges.add((c["arxiv_id"], ref_id))  # c cites ref_id
    return sorted(edges)


# --- output --------------------------------------------------------------------

def write_outputs(candidates: list[dict], edges: list[tuple[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    edge_count_by_id: dict[str, int] = {c["arxiv_id"]: 0 for c in candidates}
    for src, dst in edges:
        edge_count_by_id[src] = edge_count_by_id.get(src, 0) + 1
        edge_count_by_id[dst] = edge_count_by_id.get(dst, 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "total_internal_citation_edges": len(edges),
        "internal_edges": [{"citing": s, "cited": d} for s, d in edges],
        "candidates": candidates,
    }
    json_path = OUT_DIR / "candidates.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Candidate paper list (review before freezing)\n")
    lines.append(
        f"Generated {payload['generated_at']}. "
        f"{len(candidates)} candidates, {len(edges)} internal citation edges "
        f"(edges where BOTH endpoints are in this candidate set).\n"
    )
    lines.append(
        "**This is not the frozen corpus.** Review, drop anything off-topic or "
        "low-quality, confirm anchor papers resolved correctly, then freeze into "
        "`corpus_manifest.json` before the Phase 1 eval baseline runs.\n"
    )

    zero_edge = [c for c in candidates if edge_count_by_id.get(c["arxiv_id"], 0) == 0]
    if zero_edge:
        lines.append(
            f"⚠️ **{len(zero_edge)} of {len(candidates)} candidates have ZERO internal "
            f"citation edges** to anything else in this set — they'll be isolated nodes "
            f"in the Phase 4 graph. Consider dropping the weakest of these or adding more "
            f"anchor papers that connect to them.\n"
        )

    lines.append("| arXiv ID | Source | Year | Internal edges | Title |")
    lines.append("|---|---|---|---|---|")
    for c in sorted(candidates, key=lambda c: -edge_count_by_id.get(c["arxiv_id"], 0)):
        year = (c.get("published") or "")[:4]
        edges_n = edge_count_by_id.get(c["arxiv_id"], 0)
        flag = "🔗" if c["source"] == "anchor" else ""
        lines.append(
            f"| {c['arxiv_id']} | {c['source']} {flag} | {year} | {edges_n} | {c['title']} |"
        )

    md_path = OUT_DIR / "candidates.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"{len(candidates)} candidates, {len(edges)} internal citation edges, "
          f"{len(zero_edge)} zero-edge candidates")


# --- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-recent", type=int, default=80,
                         help="Max recent arXiv candidates to fetch before anchors (default 80)")
    parser.add_argument("--years-back", type=int, default=3,
                         help="Recency cutoff in years for the arXiv search (default 3)")
    parser.add_argument("--skip-s2", action="store_true",
                         help="Skip Semantic Scholar citation lookups (fast, but no edge data)")
    args = parser.parse_args()

    print("Fetching recent candidates from arXiv...", file=sys.stderr)
    recent = fetch_recent_candidates(args.max_recent, args.years_back)
    print(f"  got {len(recent)} recent candidates", file=sys.stderr)

    print("Fetching anchor papers from arXiv...", file=sys.stderr)
    anchors = fetch_anchor_candidates()
    print(f"  got {len(anchors)} anchor papers", file=sys.stderr)

    recent_ids = {c["arxiv_id"] for c in recent}
    anchors = [a for a in anchors if a["arxiv_id"] not in recent_ids]
    candidates = recent + anchors

    edges: list[tuple[str, str]] = []
    if not args.skip_s2:
        print("Looking up citation data on Semantic Scholar...", file=sys.stderr)
        annotate_with_citations(candidates)
        edges = compute_internal_edges(candidates)
    else:
        for c in candidates:
            c["s2_found"] = None
            c["ref_arxiv_ids"] = []
            c["citation_count"] = None

    write_outputs(candidates, edges)


if __name__ == "__main__":
    main()
