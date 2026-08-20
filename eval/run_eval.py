"""Phase 1 eval harness entry point.

    python eval/run_eval.py --provider groq --variant dense_baseline \
        --golden-set eval/data/pilot_draft.json

Order of operations is load-bearing: paths are resolved against REPO_ROOT
BEFORE bootstrapping, because bootstrap chdir's into server/ and every relative
path after that point resolves somewhere else entirely.

Prints the PROJECT_LOG.md table row to stdout rather than editing the file --
recording a result is a deliberate act, not a side effect of running a script.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval._bootstrap import bootstrap_server_context  # noqa: E402


def library_versions() -> dict:
  from importlib.metadata import PackageNotFoundError, version
  out = {}
  for pkg in ["pypdf", "langchain-text-splitters", "langchain-community",
              "chromadb", "sentence-transformers", "torch"]:
    try:
      out[pkg] = version(pkg)
    except PackageNotFoundError:
      out[pkg] = None
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--provider", default="groq")
  ap.add_argument("--golden-set", default="eval/data/golden_set.v1.json")
  ap.add_argument("--variant", default="dense_baseline")
  ap.add_argument("--retriever", choices=["dense", "hybrid"], default="dense",
                  help="which retrieval path to measure (Phase 1 dense baseline, or "
                       "Phase 2 dense+BM25+RRF)")
  ap.add_argument("--rrf-k", type=int, default=None,
                  help="hybrid only: RRF constant. Defaults to 60 (Cormack et al. 2009), "
                       "chosen a priori. Overriding it is for the labelled sensitivity "
                       "sweep ONLY -- picking whichever value scores best against this "
                       "golden set is tuning on the test set.")
  ap.add_argument("--pool", type=int, default=None,
                  help="hybrid only: per-arm candidate depth before fusion")
  ap.add_argument("--weights", type=float, nargs=2, metavar=("DENSE", "BM25"),
                  default=None,
                  help="hybrid only: weighted RRF. EXPLORATORY -- the shipped pipeline is "
                       "the unweighted formula; label any table produced with this")
  ap.add_argument("--retrieval-depth", type=int, default=50)
  ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
  ap.add_argument("--rr-cap", type=int, default=50)
  ap.add_argument("--limit", type=int, help="only score the first N questions")
  ap.add_argument("--qid", nargs="+", help="only score these qids")
  ap.add_argument("--allow-unknown-gold", action="store_true",
                  help="downgrade UNKNOWN_CHUNK_ID to a warning and EXCLUDE those "
                       "questions from all aggregates (never scores them as misses)")
  ap.add_argument("--skip-fingerprint-check", action="store_true")
  ap.add_argument("--out", default="eval/results")
  args = ap.parse_args(argv)

  # resolve BEFORE chdir
  golden_path = (REPO_ROOT / args.golden_set).resolve()
  out_root = (REPO_ROOT / args.out).resolve()
  manifest_path = (REPO_ROOT / "data" / "papers" / "corpus_manifest.json").resolve()

  bootstrap_server_context()

  from core.chunk_ids import corpus_fingerprint
  from eval.golden_set import Severity, has_errors, load_golden_set, validate_golden_set
  from eval.metrics import aggregate_by_stratum, score_question
  from eval.report import print_failures, print_report, project_log_row, write_artifacts
  from eval.retrieval import (
    DenseChromaRetriever,
    HybridRRFRetriever,
    assert_deterministic_retrieval,
    count_near_ties,
    found_by,
    load_collection_index,
    read_ingest_manifest,
    to_ranking,
  )

  # ---- collection ------------------------------------------------------
  known_ids, sha_by_id, _text = load_collection_index(args.provider)
  if not known_ids:
    print(f"ERROR: the {args.provider} collection is empty. Ingest it first:\n"
          f"  python scripts/ingest_corpus.py --provider {args.provider}",
          file=sys.stderr)
    return 2
  print(f"Collection: {len(known_ids)} chunks ({args.provider})")

  ingest_manifest = read_ingest_manifest(args.provider)
  collection_fingerprint = (ingest_manifest or {}).get("corpus_fingerprint")
  live_fingerprint = corpus_fingerprint(manifest_path)
  if collection_fingerprint and collection_fingerprint != live_fingerprint:
    print(f"ERROR: the frozen corpus has changed since ingest.\n"
          f"  collection: {collection_fingerprint}\n  corpus now: {live_fingerprint}\n"
          f"  Re-ingest before measuring.", file=sys.stderr)
    if not args.skip_fingerprint_check:
      return 2

  # ---- golden set ------------------------------------------------------
  gs = load_golden_set(golden_path)
  issues = validate_golden_set(
    gs,
    known_chunk_ids=known_ids,
    known_text_sha8=sha_by_id,
    collection_fingerprint=None if args.skip_fingerprint_check else collection_fingerprint,
    reporting_ks=args.ks,
  )

  excluded_qids = set()
  fatal = []
  for issue in issues:
    if issue.severity is Severity.ERROR and issue.code == "UNKNOWN_CHUNK_ID" and args.allow_unknown_gold:
      excluded_qids.add(issue.qid)
      print(f"WARN  (excluded) {issue}")
      continue
    if issue.severity is Severity.ERROR:
      fatal.append(issue)
    print(issue)

  if fatal:
    print(f"\nERROR: {len(fatal)} validation error(s); refusing to produce a number.",
          file=sys.stderr)
    return 2
  if excluded_qids:
    print(f"\nExcluding {len(excluded_qids)} question(s) with unreachable gold "
          f"from ALL aggregates (not scored as misses).")

  questions = [q for q in gs.questions if q.qid not in excluded_qids]
  if args.qid:
    questions = [q for q in questions if q.qid in set(args.qid)]
  if args.limit:
    questions = questions[:args.limit]
  if not questions:
    print("ERROR: no questions to score.", file=sys.stderr)
    return 2

  # ---- retrieve + score ------------------------------------------------
  if args.retriever == "hybrid":
    weights = ({"dense": args.weights[0], "bm25": args.weights[1]}
               if args.weights else None)
    retriever = HybridRRFRetriever(args.provider, variant=args.variant,
                                   rrf_k=args.rrf_k, pool=args.pool, weights=weights)
    print(f"Retriever: {retriever.name} {retriever.stats}")
  else:
    if args.rrf_k or args.pool or args.weights:
      print("ERROR: --rrf-k/--pool/--weights are hybrid-only; --retriever dense ignores "
            "them, and a run whose recorded meta implies fusion settings it never used is "
            "worse than no run.", file=sys.stderr)
      return 2
    retriever = DenseChromaRetriever(args.provider, variant=args.variant)
  deterministic = assert_deterministic_retrieval(retriever, questions[0].question,
                                                 k=args.retrieval_depth)

  scores = []
  extra = {}
  total_ties = 0
  for i, q in enumerate(questions, start=1):
    hits = retriever.retrieve(q.question, args.retrieval_depth)
    ranking = to_ranking(hits)
    ties = count_near_ties(hits, k=max(args.ks))
    total_ties += ties

    s = score_question(q.qid, q.semantics, ranking, q.group_sets,
                       ks=args.ks, rr_cap=args.rr_cap)
    scores.append(s)
    by_id = {h.chunk_id: h for h in hits}
    gold_hits = {
      cid: {
        "rank": by_id[cid].rank,
        "found_by": found_by(by_id[cid]),
        "dense_rank": by_id[cid].ranks.get("dense"),
        "sparse_rank": by_id[cid].ranks.get("bm25"),
      }
      for cid in q.all_chunk_ids if cid in by_id
    }

    extra[q.qid] = {
      "question": q.question,
      "gold_chunk_ids": list(q.all_chunk_ids),
      "provenance": q.provenance,
      "top10": ranking[:10],
      "top10_distances": [None if h.l2_distance is None else round(h.l2_distance, 4)
                          for h in hits[:10]],
      "top10_found_by": [found_by(h) for h in hits[:10]],
      "near_ties_in_topk": ties,
      # Which arm actually surfaced each gold chunk that was retrieved at all.
      # This is what separates "hybrid scored higher" from "BM25 reached evidence
      # dense never reached", and compare_runs.py aggregates it.
      "gold_hits": gold_hits,
    }
    print(f"[{i}/{len(questions)}] {q.qid}  RR={s.rr:.3f}  ranks={s.coverage_ranks}")

  strata = aggregate_by_stratum(scores)

  run_meta = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "variant": args.variant,
    "provider": args.provider,
    "retriever": retriever.name,
    "golden_set": str(golden_path.relative_to(REPO_ROOT)),
    "golden_set_version": gs.version,
    "n_questions": len(questions),
    "n_excluded": len(excluded_qids),
    "retrieval_depth": args.retrieval_depth,
    "ks": args.ks,
    "rr_cap": args.rr_cap,
    "corpus_fingerprint": live_fingerprint,
    "collection_count": len(known_ids),
    "retrieval_deterministic": deterministic,
    "total_near_ties_in_topk": total_ties,
    "retriever_stats": dict(retriever.stats) if hasattr(retriever, "stats") else None,
    "ingest_manifest": ingest_manifest,
    "library_versions": library_versions(),
  }

  print_report(strata, args.ks, run_meta)
  print_failures(scores, {q.qid: q for q in questions})

  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  out_dir = out_root / f"{stamp}_{args.variant}"
  write_artifacts(out_dir, run_meta, scores, strata, extra)
  print(f"\nArtifacts: {out_dir}")

  print("\nPROJECT_LOG.md Eval Results row (copy manually -- not auto-written):")
  print(project_log_row(args.variant, strata["all"],
                        notes=f"n={len(questions)}, {args.provider}/MiniLM-L12-v2, "
                              f"depth={args.retrieval_depth}, variant `{args.variant}`"))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
