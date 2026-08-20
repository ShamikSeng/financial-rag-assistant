"""Retrieval metrics over equivalence groups.

PURITY CONSTRAINT: this module imports nothing outside the stdlib. No LangChain,
no Chroma, no path/cwd manipulation. That is what lets its tests run in
milliseconds with no model load, which in turn is the only reason the
`all_required` scoring path is testable at all -- the bulk-generated pilot set
contains no multi-hop questions, so without synthetic fixtures that code path
would ship unexercised.

The relevance model
-------------------
Gold evidence for a question is a list of **disjoint equivalence groups**.
Credit accrues once per group, the first time any member of that group is
retrieved:

  - `any_of`      -> ONE group holding every gold id. These are alternatives
                     (adjacent chunks sharing 50 tokens of overlap that both
                     contain the answer). Retrieving one is full success;
                     retrieving a second is redundant, not extra credit.
  - `all_required`-> N SINGLETON groups. A multi-hop question where the answer
                     genuinely requires combining distinct evidence.
  - general        -> N groups of several ids each: a cross-paper multi-hop
                     question where each hop also has adjacent alternatives.
                     The flat two-valued flag cannot express this; groups can.

This is not a bespoke invention. Group recall is **S-recall / subtopic recall**
(Zhai, Cohen & Lafferty 2003) and the nDCG gain scheme is **alpha-nDCG with
alpha = 1** (Clarke et al. 2008), where a group is a "nugget". Pinning alpha = 1
over *disjoint* groups is what gives IDCG a closed form and makes nDCG <= 1
provable -- general alpha-nDCG needs a greedy approximation of an NP-hard ideal
ranking and can exceed 1. Disjointness is therefore load-bearing, not tidiness,
and is enforced rather than assumed.

Everything reduces to a single intermediate quantity: the **coverage rank**
    c_j = min { i : ranking[i] in group_j }        (INF if never covered)
Every metric below is a function of {c_1 .. c_N} alone.
"""

import math
import random
from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Optional, Sequence, Tuple

Ranking = Sequence[str]
Groups = Sequence[AbstractSet[str]]

INF = math.inf

# Depth beyond which a coverage rank counts as "not found". Recorded in every
# run's metadata, since MRR is not comparable across different caps.
DEFAULT_RR_CAP = 50
DEFAULT_KS: Tuple[int, ...] = (3, 5, 10)


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------

def validate_groups(groups: Groups) -> None:
  """Reject group structures that would silently produce wrong numbers.

  Overlap is the dangerous one: if a chunk belongs to two groups it covers both
  at a single rank, the finite coverage ranks stop being distinct, and the
  DCG <= IDCG argument collapses -- nDCG can then exceed 1. It is also
  semantically incoherent for `all_required` ("the answer needs *distinct*
  evidence" contradicted by one chunk satisfying two hops).
  """
  if not groups:
    raise ValueError("groups must be non-empty (a question needs at least one gold group)")

  seen: Dict[str, int] = {}
  for idx, group in enumerate(groups):
    if not group:
      raise ValueError(f"group {idx} is empty; every gold group needs at least one chunk id")
    for chunk_id in group:
      if chunk_id in seen:
        raise ValueError(
          f"groups must be pairwise disjoint: {chunk_id!r} appears in both "
          f"group {seen[chunk_id]} and group {idx}. Overlapping groups break the "
          f"DCG <= IDCG bound and would allow nDCG > 1."
        )
      seen[chunk_id] = idx


def dedupe_ranking(ranking: Ranking) -> List[str]:
  """First occurrence wins. A duplicate id later in the list is a no-op anyway
  (its group is already covered), so deduping changes no metric -- it just makes
  rank positions mean what they appear to mean."""
  seen = set()
  out = []
  for chunk_id in ranking:
    if chunk_id not in seen:
      seen.add(chunk_id)
      out.append(chunk_id)
  return out


# --------------------------------------------------------------------------
# the one intermediate quantity
# --------------------------------------------------------------------------

def coverage_ranks(ranking: Ranking, groups: Groups) -> List[float]:
  """1-indexed rank at which each group is first covered; INF if never."""
  validate_groups(groups)
  ordered = dedupe_ranking(ranking)

  out: List[float] = []
  for group in groups:
    rank: float = INF
    for i, chunk_id in enumerate(ordered, start=1):
      if chunk_id in group:
        rank = i
        break
    out.append(rank)
  return out


# --------------------------------------------------------------------------
# MRR family
# --------------------------------------------------------------------------

def reciprocal_rank_coverage(ranking: Ranking, groups: Groups,
                             cap: Optional[int] = DEFAULT_RR_CAP) -> float:
  """HEADLINE MRR. Reciprocal of the depth at which the question becomes answerable.

  RR = 1 / max_j c_j -- i.e. the rank at which the LAST required group arrives,
  because that is the depth at which top-d finally contains all the evidence.
  Since the chain stuffs the top-k into one prompt, this measures the quantity
  the generator actually needs rather than a proxy for it.

  On single-group questions (which is nearly all of a bulk-generated set) this
  is *identically* textbook MRR, so the logged number stays comparable to
  published MRR figures rather than being a private quantity.
  """
  ranks = coverage_ranks(ranking, groups)
  worst = max(ranks)
  if worst == INF or (cap is not None and worst > cap):
    return 0.0
  return 1.0 / worst


def reciprocal_rank_mean_groups(ranking: Ranking, groups: Groups,
                                cap: Optional[int] = DEFAULT_RR_CAP) -> float:
  """DIAGNOSTIC ONLY -- never report as the headline.

  Mean of per-group reciprocal ranks. Its failure mode is subtle and worse for
  being plausible: on a 2-hop question that finds hop A at rank 1 and never
  finds hop B, it returns 0.5 for a question the system fundamentally cannot
  answer. Half credit for zero answerability.

  Kept because it retains a *gradient* that RR_cov deliberately discards -- it
  moves when hybrid retrieval drags an uncovered hop from rank 40 to rank 12,
  which is genuinely useful when debugging why a phase did or didn't help.
  """
  ranks = coverage_ranks(ranking, groups)
  total = 0.0
  for rank in ranks:
    if rank != INF and (cap is None or rank <= cap):
      total += 1.0 / rank
  return total / len(ranks)


def reciprocal_rank_first(ranking: Ranking, groups: Groups,
                          cap: Optional[int] = DEFAULT_RR_CAP) -> float:
  """DIAGNOSTIC ONLY -- and the one to never be tempted by.

  Reciprocal rank of the FIRST covered group. On a 2-hop question that nails hop
  A at rank 1 and never finds hop B, this returns 1.0 -- a perfect score for a
  question the system cannot answer. Computed solely so the leniency gap is
  visible in the per-question dump rather than hypothetical.
  """
  ranks = coverage_ranks(ranking, groups)
  best = min(ranks)
  if best == INF or (cap is not None and best > cap):
    return 0.0
  return 1.0 / best


# --------------------------------------------------------------------------
# recall family
# --------------------------------------------------------------------------

def group_recall_at_k(ranking: Ranking, groups: Groups, k: int) -> float:
  """S-recall / subtopic recall: fraction of gold groups covered within top-k.

  At N=1 this is binary hit-rate -- retrieving 2 of 3 equivalent alternatives
  scores 1.0, not 0.667, which is the whole point of the equivalence-group
  model. At N>1 the partial credit is over *independent* nuggets, so it is
  correct rather than lenient; the metric that refuses partial credit is
  sufficiency_at_k, which is why both are reported.
  """
  if k < 1:
    raise ValueError(f"k must be >= 1, got {k}")
  ranks = coverage_ranks(ranking, groups)
  return sum(1 for r in ranks if r <= k) / len(ranks)


def sufficiency_at_k(ranking: Ranking, groups: Groups, k: int) -> float:
  """1.0 iff EVERY gold group is covered within top-k, else 0.0.

  The only metric here that answers "could the system have answered this at
  all". nDCG cannot express answerability -- a 2-hop question with hop A at
  rank 1 and hop B never retrieved still scores ~0.61 nDCG. Without this
  column, an all_required stratum showing 0.61 nDCG next to 0.00 MRR looks
  self-contradictory when in fact both are right.
  """
  if k < 1:
    raise ValueError(f"k must be >= 1, got {k}")
  ranks = coverage_ranks(ranking, groups)
  return 1.0 if all(r <= k for r in ranks) else 0.0


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------

def _discount(rank: float) -> float:
  return 1.0 / math.log2(rank + 1.0)


def dcg_at_k(ranking: Ranking, groups: Groups, k: int) -> float:
  """Gain 1 at the rank a group is first covered, 0 for redundant members.

  Because gain is nonzero at exactly one rank per group, DCG decomposes into a
  sum over groups -- no per-rank loop needed.
  """
  if k < 1:
    raise ValueError(f"k must be >= 1, got {k}")
  ranks = coverage_ranks(ranking, groups)
  return sum(_discount(r) for r in ranks if r <= k)


def idcg_at_k(n_groups: int, k: int) -> float:
  """Ideal DCG: cover a fresh group at every rank, truncated to length k.

  The truncation at min(N, k) is what keeps the metric comparable across
  questions with different group counts. Using N alone would mean a 3-group
  question can never reach nDCG@2 = 1.0 even under a perfect ranking (it would
  cap at 0.7653), so questions of different N would sit on different scales.
  """
  if n_groups < 1:
    raise ValueError(f"n_groups must be >= 1, got {n_groups}")
  if k < 1:
    raise ValueError(f"k must be >= 1, got {k}")
  return sum(_discount(i) for i in range(1, min(n_groups, k) + 1))


def ndcg_at_k(ranking: Ranking, groups: Groups, k: int) -> float:
  """nDCG in [0, 1].

  Upper bound proof (and the reason validate_groups rejects overlap): groups are
  disjoint and a retrieved chunk lies in at most one group, so the finite
  coverage ranks are DISTINCT positive integers. 1/log2(c+1) is strictly
  decreasing in c, so the sum is maximized by assigning the smallest distinct
  ranks 1..min(N,k) -- which is exactly IDCG@k. Hence DCG <= IDCG.
  """
  ideal = idcg_at_k(len(groups), k)
  if ideal == 0.0:
    return 0.0
  return dcg_at_k(ranking, groups, k) / ideal


# --------------------------------------------------------------------------
# per-question and aggregate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionScores:
  qid: str
  semantics: str
  n_groups: int
  coverage_ranks: Tuple[float, ...]
  rr: float                  # headline (RR_cov)
  rr_mean_groups: float      # diagnostic
  rr_first: float            # diagnostic
  recall_at: Mapping[int, float]
  sufficiency_at: Mapping[int, float]
  ndcg_at: Mapping[int, float]


def score_question(qid: str, semantics: str, ranking: Ranking, groups: Groups,
                   ks: Sequence[int] = DEFAULT_KS,
                   rr_cap: Optional[int] = DEFAULT_RR_CAP) -> QuestionScores:
  return QuestionScores(
    qid=qid,
    semantics=semantics,
    n_groups=len(groups),
    coverage_ranks=tuple(coverage_ranks(ranking, groups)),
    rr=reciprocal_rank_coverage(ranking, groups, cap=rr_cap),
    rr_mean_groups=reciprocal_rank_mean_groups(ranking, groups, cap=rr_cap),
    rr_first=reciprocal_rank_first(ranking, groups, cap=rr_cap),
    recall_at={k: group_recall_at_k(ranking, groups, k) for k in ks},
    sufficiency_at={k: sufficiency_at_k(ranking, groups, k) for k in ks},
    ndcg_at={k: ndcg_at_k(ranking, groups, k) for k in ks},
  )


@dataclass(frozen=True)
class AggregateScores:
  n: int
  mrr: float
  mrr_mean_groups: float
  mrr_first: float
  recall_at: Mapping[int, float]
  sufficiency_at: Mapping[int, float]
  ndcg_at: Mapping[int, float]


def _mean(values: Sequence[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def aggregate(scores: Sequence[QuestionScores]) -> AggregateScores:
  """MACRO-average: score each question to [0,1], then take an unweighted mean.

  Never micro-average (pooling groups across questions as
  total_covered / total_groups). Micro weights a 3-hop question 3x a single-hop
  one, so a handful of multi-hop questions can swing a 90-question headline.
  """
  if not scores:
    return AggregateScores(0, 0.0, 0.0, 0.0, {}, {}, {})

  ks = sorted(scores[0].recall_at.keys())
  return AggregateScores(
    n=len(scores),
    mrr=_mean([s.rr for s in scores]),
    mrr_mean_groups=_mean([s.rr_mean_groups for s in scores]),
    mrr_first=_mean([s.rr_first for s in scores]),
    recall_at={k: _mean([s.recall_at[k] for s in scores]) for k in ks},
    sufficiency_at={k: _mean([s.sufficiency_at[k] for s in scores]) for k in ks},
    ndcg_at={k: _mean([s.ndcg_at[k] for s in scores]) for k in ks},
  )


def aggregate_by_stratum(scores: Sequence[QuestionScores]) -> Dict[str, AggregateScores]:
  """Always report 'all' PLUS each semantics stratum separately.

  Pooling alone hides the signal that matters most. If ~10 multi-hop questions
  go from unanswerable to answerable, the all_required stratum shows a step
  change while the pooled headline moves by an amount indistinguishable from
  noise at n~90 -- i.e. Phase 4 could build the graph path, make it work, and
  report a delta a skeptical reader would rightly dismiss.
  """
  out = {"all": aggregate(scores)}
  for stratum in ("any_of", "all_required"):
    subset = [s for s in scores if s.semantics == stratum]
    if subset:
      out[stratum] = aggregate(subset)
  return out


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], iters: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
  """Percentile bootstrap CI for a mean."""
  if not values:
    return (0.0, 0.0)
  rng = random.Random(seed)
  n = len(values)
  means = []
  for _ in range(iters):
    means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
  means.sort()
  lo = means[int((alpha / 2) * iters)]
  hi = means[min(int((1 - alpha / 2) * iters), iters - 1)]
  return (lo, hi)


def paired_delta_ci(baseline: Sequence[float], candidate: Sequence[float],
                    iters: int = 10_000, seed: int = 0,
                    alpha: float = 0.05) -> Tuple[float, float, float]:
  """Paired bootstrap on the per-question delta. Returns (mean_delta, lo, hi).

  Paired because both systems are evaluated on the SAME questions, so the
  question-difficulty variance cancels and the interval is far tighter than two
  independent CIs would give. "Hybrid +0.041 MRR, 95% CI [+0.012, +0.071],
  n=90 paired" is a categorically stronger claim than "+0.041", and it is the
  direct answer to "how do you know that isn't noise".
  """
  if len(baseline) != len(candidate):
    raise ValueError(f"paired inputs must align: {len(baseline)} vs {len(candidate)}")
  deltas = [c - b for b, c in zip(baseline, candidate)]
  lo, hi = bootstrap_ci(deltas, iters=iters, seed=seed, alpha=alpha)
  return (_mean(deltas), lo, hi)
