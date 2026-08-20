"""Paired comparison of two eval runs, and the Phase 2 decision rule.

    python eval/compare_runs.py eval/results/<baseline_dir> eval/results/<candidate_dir>

Why this exists separately from run_eval.py
-------------------------------------------
A single run prints "nDCG@10 = 0.418". Two runs print two numbers, and the
difference between them is the entire claim a phase makes. At n=89 that
difference can easily be noise, so the honest form of the claim is a paired
bootstrap interval: "+0.041, 95% CI [+0.012, +0.071], n=89 paired". Paired
because both systems answer the SAME questions, so per-question difficulty
cancels and the interval is far tighter than two independent CIs.

`eval.metrics.paired_delta_ci` has existed since Phase 1 and was written for
exactly this moment. This module is only the plumbing around it.

Deliberately pure: reads run artifacts off disk and imports nothing but
eval.metrics, so it never needs a vectorstore and never re-runs retrieval.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import paired_delta_ci  # noqa: E402

BAR = "=" * 78

# The metric the Phase 2 default-flip rule is decided on, fixed in advance.
# Everything else here is reported but does not vote -- picking the winner from
# whichever metric happened to move is how a null result becomes a positive one.
PRIMARY = ("ndcg_at", 10)

REPORTED = [
  ("rr", None, "MRR (RR_cov)"),
  ("recall_at", 5, "Recall@5"),
  ("sufficiency_at", 5, "Sufficiency@5"),
  ("ndcg_at", 10, "nDCG@10"),
]


def load_run(path: Path) -> Dict[str, dict]:
  rows = {}
  with (path / "per_question.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
      row = json.loads(line)
      rows[row["qid"]] = row
  if not rows:
    raise SystemExit(f"No per-question rows in {path}")
  return rows


def read_meta(path: Path) -> dict:
  meta_path = path / "run_meta.json"
  return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def value(row: Mapping, field: str, k) -> float:
  if k is None:
    return float(row[field])
  return float(row[field][str(k)])


def paired_values(base: Mapping[str, dict], cand: Mapping[str, dict],
                  qids: Sequence[str], field: str, k) -> tuple:
  return ([value(base[q], field, k) for q in qids],
          [value(cand[q], field, k) for q in qids])


def mean(values: Sequence[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def compare_stratum(name: str, base: Mapping[str, dict], cand: Mapping[str, dict],
                    qids: Sequence[str], iters: int, seed: int) -> Dict[str, tuple]:
  print(f"\n  {name}  (n={len(qids)})")
  print(f"    {'metric':<16} {'baseline':>9} {'candidate':>10} {'delta':>9}   95% CI")
  out = {}
  for field, k, label in REPORTED:
    b, c = paired_values(base, cand, qids, field, k)
    delta, lo, hi = paired_delta_ci(b, c, iters=iters, seed=seed)
    out[(field, k)] = (delta, lo, hi)
    flag = "" if lo <= 0 <= hi else ("  *" if delta > 0 else "  * (regression)")
    print(f"    {label:<16} {mean(b):>9.4f} {mean(c):>10.4f} {delta:>+9.4f}   "
          f"[{lo:+.4f}, {hi:+.4f}]{flag}")
  return out


def arm_attribution(cand: Mapping[str, dict], qids: Sequence[str]) -> None:
  """Which arm actually reached the gold evidence.

  The number that matters for Phase 3 planning is `bm25 only`: gold chunks the
  dense arm never surfaced at any depth. Those are evidence a reranker could
  not previously have seen, no matter how good it is -- a reranker only reorders
  what retrieval already found.
  """
  counts = {"both": 0, "dense": 0, "bm25": 0}
  questions_rescued = 0
  unattributed = 0

  for qid in qids:
    hits = cand[qid].get("gold_hits") or {}
    rescued = False
    for detail in hits.values():
      label = detail.get("found_by")
      if label in counts:
        counts[label] += 1
      else:
        unattributed += 1
      if label == "bm25":
        rescued = True
    questions_rescued += 1 if rescued else 0

  total = sum(counts.values())
  if not total:
    print("\n  (no per-arm attribution in the candidate run -- single-arm retriever)")
    return

  print(f"\n  Gold chunks retrieved, by arm (n={total} gold chunks reached):")
  for label in ("both", "dense", "bm25"):
    share = counts[label] / total * 100
    print(f"    {label:<12} {counts[label]:>4}  ({share:4.1f}%)")
  if unattributed:
    print(f"    {'(none)':<12} {unattributed:>4}")
  print(f"\n    {questions_rescued} question(s) had gold evidence that ONLY the BM25 arm "
        f"reached.\n    Those are the ones dense-only retrieval could not have answered at "
        f"any depth.")


def movement(base: Mapping[str, dict], cand: Mapping[str, dict],
             qids: Sequence[str], limit: int = 10) -> None:
  deltas = [(value(cand[q], "rr", None) - value(base[q], "rr", None), q) for q in qids]
  improved = sorted([d for d in deltas if d[0] > 1e-9], reverse=True)
  worsened = sorted([d for d in deltas if d[0] < -1e-9])
  unchanged = len(deltas) - len(improved) - len(worsened)

  print(f"\n  Per-question RR movement: {len(improved)} up, {len(worsened)} down, "
        f"{unchanged} unchanged")
  rescued = [q for q in qids
             if value(base[q], "rr", None) == 0.0 and value(cand[q], "rr", None) > 0.0]
  lost = [q for q in qids
          if value(base[q], "rr", None) > 0.0 and value(cand[q], "rr", None) == 0.0]
  print(f"    total misses rescued: {len(rescued)}   previously-found now missed: {len(lost)}")
  if lost:
    print(f"      newly missed: {', '.join(lost[:limit])}")


def verdict(pooled: Dict[str, tuple], strata: Dict[str, Dict[str, tuple]]) -> bool:
  """The rule pre-registered in PROJECT_LOG.md, applied mechanically.

  Stated before the numbers were seen, and evaluated here without discretion --
  the point of pre-registering it is that it cannot be reinterpreted now that
  the result is visible.
  """
  delta, lo, hi = pooled[PRIMARY]
  improves = delta > 0 and lo > 0

  regressions = []
  for name, table in strata.items():
    if name == "all":
      continue
    s_delta, s_lo, s_hi = table[PRIMARY]
    if s_hi < 0:
      regressions.append(f"{name} (delta {s_delta:+.4f}, CI upper {s_hi:+.4f})")

  print("\n" + BAR)
  print("PRE-REGISTERED DECISION RULE (PROJECT_LOG.md, fixed before this run)")
  print(BAR)
  print(f"  primary metric        pooled nDCG@10")
  print(f"  mean paired delta     {delta:+.4f}   (must be > 0)          "
        f"{'PASS' if delta > 0 else 'FAIL'}")
  print(f"  95% CI lower bound    {lo:+.4f}   (must be > 0)          "
        f"{'PASS' if lo > 0 else 'FAIL'}")
  print(f"  no stratum regresses  {'none' if not regressions else '; '.join(regressions)}"
        f"          {'PASS' if not regressions else 'FAIL'}")
  fires = improves and not regressions
  print()
  if fires:
    print("  VERDICT: rule FIRES -- the candidate becomes the app default")
    print("           (set DEFAULT_RETRIEVAL_MODE in server/config/settings.py)")
  else:
    print("  VERDICT: rule does NOT fire -- the default stays unchanged.")
    print("           Record the finding anyway; a null result measured properly is")
    print("           still a result, and reinterpreting the rule now would defeat")
    print("           the reason it was written down first.")
  print(BAR)
  return fires


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("baseline", help="eval/results/<dir> of the baseline run")
  ap.add_argument("candidate", help="eval/results/<dir> of the candidate run")
  ap.add_argument("--iters", type=int, default=10_000)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)

  base_dir = (REPO_ROOT / args.baseline).resolve()
  cand_dir = (REPO_ROOT / args.candidate).resolve()
  base, cand = load_run(base_dir), load_run(cand_dir)
  base_meta, cand_meta = read_meta(base_dir), read_meta(cand_dir)

  print(BAR)
  print("PAIRED RUN COMPARISON")
  print(BAR)
  print(f"  baseline    {base_meta.get('variant', '?')}  ({base_dir.name})")
  print(f"  candidate   {cand_meta.get('variant', '?')}  ({cand_dir.name})")

  base_gs = base_meta.get("golden_set_version")
  cand_gs = cand_meta.get("golden_set_version")
  print(f"  golden set  {base_gs} vs {cand_gs}")
  if base_gs != cand_gs:
    print("\nERROR: the two runs were scored against different golden set versions.\n"
          "  Rows are only comparable within one version (PROJECT_LOG.md comparability\n"
          "  rule) -- re-score the baseline against the new version first.", file=sys.stderr)
    return 2

  common = [q for q in base if q in cand]
  if len(common) != len(base) or len(common) != len(cand):
    print(f"\nWARNING: question sets differ -- baseline {len(base)}, candidate "
          f"{len(cand)}, comparing the {len(common)} in both.")
  if not common:
    print("ERROR: no questions in common.", file=sys.stderr)
    return 2
  common.sort()

  strata: Dict[str, Dict[str, tuple]] = {}
  by_semantics: Dict[str, List[str]] = {}
  for qid in common:
    by_semantics.setdefault(base[qid]["semantics"], []).append(qid)

  print(f"\n{BAR}\nPAIRED DELTAS ({args.iters} bootstrap iters, seed {args.seed})\n{BAR}")
  strata["all"] = compare_stratum("all", base, cand, common, args.iters, args.seed)
  for name in ("any_of", "all_required"):
    if by_semantics.get(name):
      strata[name] = compare_stratum(name, base, cand, by_semantics[name],
                                     args.iters, args.seed)
  print("\n  * = interval excludes zero")

  print(f"\n{BAR}\nDIAGNOSTICS\n{BAR}")
  movement(base, cand, common)
  arm_attribution(cand, common)

  fires = verdict(strata["all"], strata)
  return 0 if fires else 1


if __name__ == "__main__":
  raise SystemExit(main())
