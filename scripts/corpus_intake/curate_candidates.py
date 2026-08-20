"""
Applies a manual relevance curation pass to data/papers/candidates.json.

Internal citation edge count (from fetch_candidate_papers.py) is a good signal
but not sufficient on its own: some clearly on-topic papers show 0 edges only
because they're too recent for Semantic Scholar's index to have caught up, and
some off-topic papers picked up nonzero edges just by citing a widely-cited
anchor in passing. This applies an explicit, reviewable keep-list instead of a
pure edge-count cutoff -- see the inline reasoning next to each ID.

Usage:
    python scripts/corpus_intake/curate_candidates.py
Writes:
    data/papers/curated_candidates.json
    data/papers/curated_candidates.md
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "papers"

# All 14 anchors are kept -- each was added deliberately for either graph
# connectivity or a direct phase tie-in (see fetch_candidate_papers.py).
ANCHOR_IDS = {
    "2004.04906", "2005.11401", "2002.08909", "1901.04085", "2004.12832",
    "2404.16130", "2307.03172", "2310.11511", "2112.01488", "2309.15217",
    "2401.15884", "2402.03216", "2403.14403", "2407.01449",
}

# Recent-pool papers kept after a topical relevance read of all 80 titles,
# not just edge count. Selection favors: (a) higher internal edge count as a
# tie-breaker, (b) coverage across every phase (hybrid, rerank, GraphRAG,
# multimodal, long-context, eval-adjacent), (c) dropping recommendation-system/
# RL-alignment/unrelated-domain papers that only matched because "cs.LG" is a
# broad category and some keyword happened to co-occur.
RECENT_KEEP_IDS = {
    "2607.24663",  # corrective agentic hybrid RAG
    "2607.24165",  # Do Current Retrievers Cover All the Evidence?
    "2607.26497",  # BM25 Wins at Scale
    "2607.24554",  # DeCoRAG
    "2607.23561",  # Towards a Relevance Posterior in Neural Information Access
    "2607.23006",  # VecTree-RAG
    "2608.00658",  # Select-And-Extract
    "2608.00585",  # Verification Without Sufficiency (multi-hop RAG)
    "2607.28397",  # GLM-RAG (graph-based RAG)
    "2607.27353",  # LayerRAG-Bench
    "2607.27136",  # KAMR (knowledge-aligned multi-hop retrieval)
    "2608.01565",  # DocNavRAG (document-structured graph RAG)
    "2607.24223",  # Guiding Corpus Interaction in Agentic Search
    "2607.22479",  # Legal Nugget Extraction for Granular Retrieval
    "2607.29402",  # Bridging the Question-Answer Gap (HyDE-style)
    "2608.02583",  # UEmbed (unified sparse/dense multimodal embeddings)
    "2608.01450",  # Real-Time Hybrid Retrieval in Hyperbolic Space
    "2608.00705",  # Triple-Robustness Analysis of RAG, Multi-Hop
    "2607.25959",  # Detecting Knowledge Inconsistencies Text/Tables/KG
    "2607.23507",  # Choosing a Text Embedding Model
    "2608.03091",  # Position Bias in Listwise LLM Reranking
    "2608.02189",  # Disentangled Contrastive Multilingual Dense Retrieval
    "2608.00765",  # RAGOCR (optical compression, multimodal RAG)
    "2607.24332",  # Cross-Attention Calibrated Deduplication for RAG
    "2607.24861",  # HVM-GraphRAG (multimodal graph RAG)
    "2608.03860",  # SciRet (retrieval/reranking for scientific RAG)
    "2608.03487",  # RAG-Stack (serving performance -- Phase 7 tie-in)
    "2608.01269",  # ACE-GraphRAG
    "2608.00916",  # Tevatron Meets Megatron (reranker training at scale)
    "2607.25182",  # TabRank (table re-rankers -- Phase 3 + 5 tie-in)
    "2608.01468",  # Retrieval-Augmented Biomedical QA (BioASQ)
    # NOTE: 2608.01732 (X-KGRank) was in an earlier pass of this list -- dropped
    # after reading the actual abstract, not just the title. It's a recommender-
    # systems explainability paper (KG-based reasoning for product recommendations),
    # not document retrieval for QA -- same category as the recommendation-system
    # papers already excluded elsewhere (DIRECTOR, TopoGR, CALMRec). The title alone
    # ("RAG Framework", "Knowledge Graph") reads as on-topic; the abstract doesn't.
}

KEEP_IDS = ANCHOR_IDS | RECENT_KEEP_IDS


def main() -> None:
    payload = json.loads((DATA_DIR / "candidates.json").read_text(encoding="utf-8"))
    all_candidates = payload["candidates"]

    kept = [c for c in all_candidates if c["arxiv_id"] in KEEP_IDS]
    dropped = [c for c in all_candidates if c["arxiv_id"] not in KEEP_IDS]

    kept_ids = {c["arxiv_id"] for c in kept}
    edges = [e for e in payload["internal_edges"]
             if e["citing"] in kept_ids and e["cited"] in kept_ids]

    out = {
        "generated_from": "candidates.json",
        "total_kept": len(kept),
        "total_dropped": len(dropped),
        "internal_edges_after_curation": len(edges),
        "kept": kept,
        "dropped_ids": sorted(c["arxiv_id"] for c in dropped),
    }
    (DATA_DIR / "curated_candidates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    edge_count = {c["arxiv_id"]: 0 for c in kept}
    for e in edges:
        edge_count[e["citing"]] += 1
        edge_count[e["cited"]] += 1

    lines = [
        "# Curated candidate list (final review before freezing)\n",
        f"{len(kept)} kept, {len(dropped)} dropped, {len(edges)} internal citation "
        f"edges among the kept set.\n",
        "Still not frozen -- review, then freeze into `corpus_manifest.json`.\n",
        "| arXiv ID | Source | Year | Internal edges | Title |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(kept, key=lambda c: -edge_count.get(c["arxiv_id"], 0)):
        year = (c.get("published") or "")[:4]
        flag = "🔗" if c["source"] == "anchor" else ""
        lines.append(
            f"| {c['arxiv_id']} | {c['source']} {flag} | {year} | "
            f"{edge_count.get(c['arxiv_id'], 0)} | {c['title']} |"
        )
    (DATA_DIR / "curated_candidates.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Kept {len(kept)}, dropped {len(dropped)}, "
          f"{len(edges)} internal edges among the kept set")
    print(f"Wrote {DATA_DIR / 'curated_candidates.json'}")
    print(f"Wrote {DATA_DIR / 'curated_candidates.md'}")


if __name__ == "__main__":
    main()
