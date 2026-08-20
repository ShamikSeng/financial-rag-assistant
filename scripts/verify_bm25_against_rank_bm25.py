"""Verify the hand-rolled BM25 against `rank_bm25`, the reference implementation.

`rank_bm25` is deliberately NOT a dependency of this project (see the Key
Architecture Decisions row on hand-rolling BM25). It is installed temporarily,
run once, and removed:

    pip install rank-bm25
    python scripts/verify_bm25_against_rank_bm25.py
    pip uninstall -y rank-bm25

What is actually checked
------------------------
Two things, because they answer different questions:

1. **Exact score agreement** between `core.bm25.BM25Index(idf_variant="okapi")`
   and `rank_bm25.BM25Okapi`, over the full 2131-document corpus, for every
   query. Both are fed the SAME tokenizer output, so any discrepancy is in the
   scoring arithmetic -- IDF, length normalisation, tf saturation -- and nowhere
   else. This is the part that catches the bug worth catching: a subtly wrong
   BM25 still returns plausible documents, and would quietly make the Phase 2
   hybrid numbers *worse* than they should be while looking like a fair result.

2. **Ranking agreement** of the shipped `idf_variant="lucene"` mode against the
   same reference. This is reported, not asserted equal: the Lucene IDF is a
   different (strictly positive) formula, so exact equality would mean the
   variant switch does nothing. What matters is that the two agree on what the
   top documents are.

Exits non-zero if (1) fails. That is a hard gate: no Phase 2 number should be
recorded against an unverified sparse arm.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_SET = REPO_ROOT / "eval" / "data" / "golden_set.v1.json"
N_QUERIES = 20
TOLERANCE = 1e-9


def load_queries(limit: int) -> list:
  questions = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]
  # Real questions rather than invented ones: they carry the corpus's actual
  # vocabulary, including the method names and hyphenated compounds the
  # tokenizer treats specially.
  return [q["question"] for q in questions[:limit]]


def main() -> int:
  try:
    from rank_bm25 import BM25Okapi
  except ImportError:
    print("rank_bm25 is not installed (by design -- it is not a dependency).\n"
          "To run this verification:\n"
          "    pip install rank-bm25\n"
          "    python scripts/verify_bm25_against_rank_bm25.py\n"
          "    pip uninstall -y rank-bm25", file=sys.stderr)
    return 3

  from eval._bootstrap import bootstrap_server_context
  bootstrap_server_context()

  from core.bm25 import OKAPI_EPSILON, OKAPI_K1, DEFAULT_B, BM25Index, tokenize
  from core.vector_database import load_vectorstore

  got = load_vectorstore("groq")._collection.get(include=["documents", "metadatas"])
  ids = [m.get("chunk_id") or cid for cid, m in zip(got["ids"], got["metadatas"])]
  texts = got["documents"]
  print(f"Corpus: {len(ids)} chunks")

  tokenized = [tokenize(t) for t in texts]

  ours = BM25Index(ids, texts, k1=OKAPI_K1, b=DEFAULT_B,
                   idf_variant="okapi", epsilon=OKAPI_EPSILON)
  theirs = BM25Okapi(tokenized, k1=OKAPI_K1, b=DEFAULT_B, epsilon=OKAPI_EPSILON)

  print(f"avgdl  ours={ours.avgdl:.6f}  rank_bm25={theirs.avgdl:.6f}")
  if abs(ours.avgdl - theirs.avgdl) > TOLERANCE:
    print("FAIL: average document length differs -- the two are not indexing the "
          "same thing.", file=sys.stderr)
    return 1

  shipped = BM25Index(ids, texts)     # lucene idf, shipped defaults

  queries = load_queries(N_QUERIES)
  worst = 0.0
  failures = 0
  overlap_total = 0

  for i, query in enumerate(queries, start=1):
    terms = tokenize(query)
    reference = theirs.get_scores(terms)
    mine = ours.score_all(query)

    delta = max(abs(mine.get(doc_index, 0.0) - float(reference[doc_index]))
                for doc_index in range(len(ids)))
    worst = max(worst, delta)
    if delta > TOLERANCE:
      failures += 1
      print(f"  [{i}/{len(queries)}] MISMATCH  max|delta|={delta:.3e}  {query[:70]!r}")
      continue

    ref_top = [ids[j] for j in sorted(range(len(ids)),
                                      key=lambda j: (-reference[j], ids[j]))[:10]]
    shipped_top = [cid for cid, _ in shipped.search(query, 10)]
    overlap = len(set(ref_top) & set(shipped_top))
    overlap_total += overlap
    print(f"  [{i}/{len(queries)}] exact match (max|delta|={delta:.3e})  "
          f"lucene-vs-okapi top-10 overlap {overlap}/10")

  print("\n" + "=" * 70)
  print(f"Okapi-mode exact agreement : {len(queries) - failures}/{len(queries)} queries "
        f"(worst |delta| = {worst:.3e}, tolerance {TOLERANCE:.0e})")
  print(f"Shipped lucene-mode top-10 overlap with reference: "
        f"{overlap_total / max(len(queries) - failures, 1):.1f}/10 mean")
  print("=" * 70)

  if failures:
    print("FAIL: the hand-rolled implementation does not reproduce rank_bm25.",
          file=sys.stderr)
    return 1
  print("PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
