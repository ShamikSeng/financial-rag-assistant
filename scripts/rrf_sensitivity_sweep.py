"""RRF sensitivity sweep -- a ROBUSTNESS DIAGNOSTIC, never a tuning run.

    python scripts/rrf_sensitivity_sweep.py

The shipped pipeline uses unweighted RRF at k=60, the constant from Cormack et
al. 2009, chosen before any of these numbers existed. This script exists to
answer "is that result fragile to the constant?" -- NOT to find a better one.
Picking whichever k scores highest here and reporting it as the pipeline would
be tuning a hyperparameter on the same 89 questions the headline is claimed
from, with no held-out set. The output table is labelled accordingly and the
weighted rows are marked exploratory: they are not the shipped formula.

Both arms are queried ONCE per question and their ranked lists re-fused under
each configuration. That is exact (fusion is a pure function of the two rankings)
and avoids 6x the embedding cost of re-running the whole harness per config.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Unweighted k sweep first (same family as the shipped config), then the
# weighted variants, which are a DIFFERENT formula and flagged as such.
CONFIGS = [
  ("k=10",            10,  None),
  ("k=30",            30,  None),
  ("k=60 (SHIPPED)",  60,  None),
  ("k=100",           100, None),
  ("k=60 w 2:1 dense (exploratory)", 60, {"dense": 2.0, "bm25": 1.0}),
  ("k=60 w 1:2 bm25  (exploratory)", 60, {"dense": 1.0, "bm25": 2.0}),
]


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--provider", default="groq")
  ap.add_argument("--golden-set", default="eval/data/golden_set.v1.json")
  ap.add_argument("--depth", type=int, default=50)
  ap.add_argument("--out", default="eval/results")
  args = ap.parse_args(argv)

  golden_path = (REPO_ROOT / args.golden_set).resolve()
  out_root = (REPO_ROOT / args.out).resolve()

  from eval._bootstrap import bootstrap_server_context
  bootstrap_server_context()

  from core.hybrid_retriever import DENSE_ARM, SPARSE_ARM, HybridRetriever, rrf_fuse
  from eval.golden_set import load_golden_set
  from eval.metrics import aggregate_by_stratum, score_question

  gs = load_golden_set(golden_path)
  questions = list(gs.questions)
  print(f"Golden set: {len(questions)} questions ({gs.version})")

  retriever = HybridRetriever(args.provider, pool=args.depth)
  print(f"Retriever: {retriever.stats}\n")

  # One pass over the corpus per question, shared by every configuration.
  arms = {}
  for i, q in enumerate(questions, start=1):
    dense = [cid for cid, _ in retriever.dense_arm(q.question, args.depth)]
    sparse = [cid for cid, _ in retriever.sparse_arm(q.question, args.depth)]
    arms[q.qid] = (dense, sparse)
    if i % 20 == 0 or i == len(questions):
      print(f"  retrieved {i}/{len(questions)}")

  rows = []
  for label, rrf_k, weights in CONFIGS:
    scores = []
    for q in questions:
      dense, sparse = arms[q.qid]
      fused = rrf_fuse({DENSE_ARM: dense, SPARSE_ARM: sparse}, k=rrf_k, weights=weights)
      ranking = [cid for cid, _s, _r in fused][:args.depth]
      scores.append(score_question(q.qid, q.semantics, ranking, q.group_sets,
                                   ks=[3, 5, 10], rr_cap=args.depth))
    strata = aggregate_by_stratum(scores)
    rows.append((label, rrf_k, weights, strata))

  # ---- report ----------------------------------------------------------
  bar = "=" * 78
  print("\n" + bar)
  print("RRF SENSITIVITY SWEEP -- DIAGNOSTIC ONLY, NOT A TUNING RESULT")
  print("The shipped pipeline is unweighted RRF at k=60, fixed a priori.")
  print(bar)
  print(f"{'config':<34} {'MRR':>7} {'R@5':>7} {'nDCG@10':>8}   {'any_of nDCG@10':>14}")
  for label, _k, _w, strata in rows:
    a = strata["all"]
    anyof = strata.get("any_of")
    print(f"{label:<34} {a.mrr:>7.4f} {a.recall_at[5]:>7.4f} {a.ndcg_at[10]:>8.4f}   "
          f"{(anyof.ndcg_at[10] if anyof else 0):>14.4f}")
  print(bar)

  unweighted = [r for r in rows if r[2] is None]
  spread = max(r[3]["all"].ndcg_at[10] for r in unweighted) - \
           min(r[3]["all"].ndcg_at[10] for r in unweighted)
  print(f"\nUnweighted k in {{10,30,60,100}}: pooled nDCG@10 spread = {spread:.4f}")
  print("Compare against the dense->hybrid delta to judge whether the Phase 2\n"
        "conclusion depends on the choice of k.")

  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  out_dir = out_root / f"{stamp}_rrf_sensitivity_sweep"
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "sweep.json").write_text(json.dumps({
    "note": "DIAGNOSTIC ONLY. Shipped pipeline is unweighted RRF k=60, chosen a priori. "
            "Weighted rows are a different (exploratory) formula and are not the "
            "reported pipeline.",
    "golden_set": str(golden_path.relative_to(REPO_ROOT)),
    "golden_set_version": gs.version,
    "n_questions": len(questions),
    "depth": args.depth,
    "retriever_stats": dict(retriever.stats),
    "results": [
      {
        "label": label, "rrf_k": rrf_k, "weights": weights,
        "strata": {
          name: {"n": agg.n, "mrr": agg.mrr,
                 "recall_at": {str(k): v for k, v in agg.recall_at.items()},
                 "ndcg_at": {str(k): v for k, v in agg.ndcg_at.items()},
                 "sufficiency_at": {str(k): v for k, v in agg.sufficiency_at.items()}}
          for name, agg in strata.items()
        },
      }
      for label, rrf_k, weights, strata in rows
    ],
  }, indent=2), encoding="utf-8")
  print(f"\nArtifacts: {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
