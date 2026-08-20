"""Console tables and run artifacts."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from eval.metrics import AggregateScores, QuestionScores

BAR = "=" * 78


def _fmt(value: float) -> str:
  return f"{value:.4f}"


def format_stratum(name: str, agg: AggregateScores, ks: Sequence[int]) -> str:
  lines = [f"  {name:<14} n={agg.n:<4} MRR={_fmt(agg.mrr)}"]
  lines.append("      " + "  ".join(f"R@{k}={_fmt(agg.recall_at[k])}" for k in ks))
  lines.append("      " + "  ".join(f"Suff@{k}={_fmt(agg.sufficiency_at[k])}" for k in ks))
  lines.append("      " + "  ".join(f"nDCG@{k}={_fmt(agg.ndcg_at[k])}" for k in ks))
  return "\n".join(lines)


def print_report(strata: Mapping[str, AggregateScores], ks: Sequence[int],
                 meta: Optional[Mapping[str, Any]] = None) -> None:
  print("\n" + BAR)
  print("RETRIEVAL EVAL")
  print(BAR)
  if meta:
    for key in ("variant", "provider", "golden_set_version", "n_questions",
                "retrieval_depth", "rr_cap"):
      if key in meta:
        print(f"  {key:<20} {meta[key]}")
    print()

  # 'all' first, then each stratum -- pooling is never the only number shown
  for name in ("all", "any_of", "all_required"):
    if name in strata:
      print(format_stratum(name, strata[name], ks))
      print()

  if "all_required" not in strata:
    print("  (no all_required questions in this set -- the multi-hop stratum will")
    print("   appear once the hand-written hard set is merged in)\n")

  agg = strata["all"]
  print("  diagnostics (NOT the headline -- see metrics.py for why):")
  print(f"      MRR_mean_groups={_fmt(agg.mrr_mean_groups)}   "
        f"MRR_first_group={_fmt(agg.mrr_first)}")
  print(BAR)


def print_failures(scores: Sequence[QuestionScores], questions_by_qid: Mapping[str, Any],
                   limit: int = 15) -> None:
  """The per-question misses. Worth more than the aggregate when diagnosing."""
  missed = [s for s in scores if s.rr == 0.0]
  if not missed:
    print("\nNo complete misses.")
    return
  print(f"\n{len(missed)} question(s) with RR=0 (gold never retrieved within cap):\n")
  for s in missed[:limit]:
    q = questions_by_qid.get(s.qid)
    text = getattr(q, "question", "") if q else ""
    print(f"  [{s.qid}] {text}")
    print(f"        gold: {getattr(q, 'all_chunk_ids', ())}")
    print(f"        coverage_ranks: {s.coverage_ranks}")
  if len(missed) > limit:
    print(f"  ... and {len(missed) - limit} more")


def write_artifacts(out_dir: Path, run_meta: Mapping[str, Any],
                    scores: Sequence[QuestionScores],
                    strata: Mapping[str, AggregateScores],
                    per_question_extra: Optional[Mapping[str, Any]] = None) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)

  (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str),
                                         encoding="utf-8")

  with (out_dir / "per_question.jsonl").open("w", encoding="utf-8") as fh:
    for s in scores:
      row = asdict(s)
      row["coverage_ranks"] = [None if r == float("inf") else r for r in s.coverage_ranks]
      row["recall_at"] = {str(k): v for k, v in s.recall_at.items()}
      row["sufficiency_at"] = {str(k): v for k, v in s.sufficiency_at.items()}
      row["ndcg_at"] = {str(k): v for k, v in s.ndcg_at.items()}
      if per_question_extra and s.qid in per_question_extra:
        row.update(per_question_extra[s.qid])
      fh.write(json.dumps(row, default=str) + "\n")

  summary = {
    name: {
      "n": agg.n,
      "mrr": agg.mrr,
      "mrr_mean_groups": agg.mrr_mean_groups,
      "mrr_first": agg.mrr_first,
      "recall_at": {str(k): v for k, v in agg.recall_at.items()},
      "sufficiency_at": {str(k): v for k, v in agg.sufficiency_at.items()},
      "ndcg_at": {str(k): v for k, v in agg.ndcg_at.items()},
    }
    for name, agg in strata.items()
  }
  (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def project_log_row(variant: str, agg: AggregateScores, notes: str = "") -> str:
  """The markdown row for PROJECT_LOG.md's Eval Results table (printed, not auto-written)."""
  return (f"| {variant} | {agg.mrr:.3f} | {agg.recall_at.get(5, 0):.3f} | "
          f"{agg.ndcg_at.get(10, 0):.3f} | {notes} |")
