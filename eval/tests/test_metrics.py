"""Fixtures pinning every metric decision with hand-computable expected values.

Five of these are load-bearing -- they pin choices that would otherwise drift
silently and quietly change what the Eval Results table means:

  T3  redundancy invariance (any_of)      T7  redundancy invariance (general case)
  T6  RR_cov strictness vs the lenient variants
  T9  IDCG truncation at min(N, k)
  T18 macro- not micro-averaging

Everything here is two literals and a float. No corpus, no embeddings, no cwd
games -- which is exactly why the all_required path can be tested at all, given
the bulk-generated pilot set contains no multi-hop questions.
"""

import math

import pytest

from eval.metrics import (
  aggregate,
  aggregate_by_stratum,
  coverage_ranks,
  dcg_at_k,
  group_recall_at_k,
  idcg_at_k,
  ndcg_at_k,
  paired_delta_ci,
  reciprocal_rank_coverage,
  reciprocal_rank_first,
  reciprocal_rank_mean_groups,
  score_question,
  sufficiency_at_k,
  validate_groups,
)

INF = math.inf

# hand-computed discounts
D1 = 1.0                      # 1/log2(2)
D2 = 0.6309297535714575       # 1/log2(3)
D3 = 0.5                      # 1/log2(4)
D4 = 0.43067655807339306      # 1/log2(5)


def approx(x):
  return pytest.approx(x, abs=1e-9)


# ---------------------------------------------------------------- basics ---

def test_T1_single_gold_at_rank_1():
  groups = [{"a"}]
  ranking = ["a", "x", "y"]
  assert coverage_ranks(ranking, groups) == [1]
  assert reciprocal_rank_coverage(ranking, groups) == approx(1.0)
  assert group_recall_at_k(ranking, groups, 5) == approx(1.0)
  assert sufficiency_at_k(ranking, groups, 5) == approx(1.0)
  assert ndcg_at_k(ranking, groups, 10) == approx(1.0)


def test_T2_any_of_covered_at_rank_2():
  groups = [{"a", "b"}]
  ranking = ["x", "a", "b"]
  assert coverage_ranks(ranking, groups) == [2]
  assert reciprocal_rank_coverage(ranking, groups) == approx(0.5)
  assert group_recall_at_k(ranking, groups, 5) == approx(1.0)
  assert ndcg_at_k(ranking, groups, 10) == approx(D2)


def test_T3_any_of_redundancy_invariance():
  """LOAD-BEARING. Retrieving a second equivalent chunk must change nothing.

  A naive "every gold chunk has gain 1" scheme would score [x,a,b] at 0.712
  nDCG vs 0.631 for [x,a] -- paying +0.081 for retrieving a chunk that shares
  50 tokens of overlap with one already retrieved and contains the same answer.
  """
  groups = [{"a", "b"}]
  full = ["x", "a", "b"]
  partial = ["x", "a"]

  assert coverage_ranks(full, groups) == coverage_ranks(partial, groups)
  for k in (1, 3, 5, 10):
    assert ndcg_at_k(full, groups, k) == approx(ndcg_at_k(partial, groups, k))
    assert group_recall_at_k(full, groups, k) == approx(group_recall_at_k(partial, groups, k))
  assert reciprocal_rank_coverage(full, groups) == approx(reciprocal_rank_coverage(partial, groups))
  # and specifically: NOT the inflated value
  assert ndcg_at_k(full, groups, 10) == approx(D2)
  assert ndcg_at_k(full, groups, 10) != pytest.approx(0.7122633, abs=1e-4)


def test_T4_never_covered():
  groups = [{"a"}]
  ranking = ["x", "y", "z"]
  assert coverage_ranks(ranking, groups) == [INF]
  assert reciprocal_rank_coverage(ranking, groups) == approx(0.0)
  assert group_recall_at_k(ranking, groups, 5) == approx(0.0)
  assert sufficiency_at_k(ranking, groups, 5) == approx(0.0)
  assert ndcg_at_k(ranking, groups, 10) == approx(0.0)


# ------------------------------------------------------- all_required ------

def test_T5_two_hops_both_covered():
  groups = [{"p"}, {"q"}]
  ranking = ["p", "x", "y", "q", "z"]
  assert coverage_ranks(ranking, groups) == [1, 4]

  assert reciprocal_rank_coverage(ranking, groups) == approx(0.25)
  assert reciprocal_rank_mean_groups(ranking, groups) == approx(0.625)
  assert reciprocal_rank_first(ranking, groups) == approx(1.0)

  assert group_recall_at_k(ranking, groups, 5) == approx(1.0)
  assert group_recall_at_k(ranking, groups, 3) == approx(0.5)
  assert sufficiency_at_k(ranking, groups, 5) == approx(1.0)
  assert sufficiency_at_k(ranking, groups, 3) == approx(0.0)

  assert ndcg_at_k(ranking, groups, 10) == approx((D1 + D4) / (D1 + D2))


def test_T6_leniency_gap_second_hop_missing():
  """LOAD-BEARING. Documents, in code, exactly what the rejected variants do.

  Hop A at rank 1, hop B never retrieved -- the question is unanswerable.
  RR_cov says 0.0 (correct). The mean-of-groups variant says 0.5, half credit
  for zero answerability. The first-group variant says 1.0, a perfect score.
  That gap is why the headline is RR_cov.
  """
  groups = [{"p"}, {"q"}]
  ranking = ["p", "x", "y", "z"]
  assert coverage_ranks(ranking, groups) == [1, INF]

  assert reciprocal_rank_coverage(ranking, groups) == approx(0.0)
  assert reciprocal_rank_mean_groups(ranking, groups) == approx(0.5)
  assert reciprocal_rank_first(ranking, groups) == approx(1.0)

  assert group_recall_at_k(ranking, groups, 5) == approx(0.5)
  assert sufficiency_at_k(ranking, groups, 5) == approx(0.0)
  # nDCG still reads 0.61 on an unanswerable question -- it cannot express
  # answerability, which is precisely why sufficiency is a reported column.
  assert ndcg_at_k(ranking, groups, 10) == approx(D1 / (D1 + D2))


def test_T7_general_case_redundancy_invariance():
  """LOAD-BEARING. Two hops, each with an adjacent alternative.

  Must score IDENTICALLY to T5 (two bare hops): the extra alternatives are
  redundant coverage, not extra evidence. This is the case a flat
  any_of/all_required flag cannot express at all.
  """
  groups = [{"a1", "a2"}, {"b1", "b2"}]
  ranking = ["a2", "a1", "x", "b1", "b2"]
  assert coverage_ranks(ranking, groups) == [1, 4]

  t5_groups = [{"p"}, {"q"}]
  t5_ranking = ["p", "x", "y", "q", "z"]

  assert reciprocal_rank_coverage(ranking, groups) == approx(
    reciprocal_rank_coverage(t5_ranking, t5_groups))
  assert ndcg_at_k(ranking, groups, 10) == approx(ndcg_at_k(t5_ranking, t5_groups, 10))
  assert group_recall_at_k(ranking, groups, 5) == approx(
    group_recall_at_k(t5_ranking, t5_groups, 5))


def test_T8_group_order_and_set_order_irrelevant():
  ranking = ["x", "b", "a"]
  assert coverage_ranks(ranking, [{"a", "b"}]) == coverage_ranks(ranking, [{"b", "a"}])


# --------------------------------------------------------------- IDCG ------

def test_T9_idcg_truncates_at_min_n_k():
  """LOAD-BEARING. A perfect ranking must score exactly 1.0 at any k.

  3 groups ranked perfectly at 1,2,3 with k=2: truncating IDCG at min(N,k)=2
  gives 1.0. Using all N=3 would give 0.7653, meaning a perfect system is
  unable to score 1 and questions with different N sit on different scales.
  """
  groups = [{"g1"}, {"g2"}, {"g3"}]
  ranking = ["g1", "g2", "g3"]

  assert ndcg_at_k(ranking, groups, 2) == approx(1.0)
  assert group_recall_at_k(ranking, groups, 2) == approx(2 / 3)

  assert idcg_at_k(3, 2) == approx(D1 + D2)
  assert dcg_at_k(ranking, groups, 2) == approx(D1 + D2)
  # the rejected convention, spelled out so the difference is not theoretical
  assert dcg_at_k(ranking, groups, 2) / (D1 + D2 + D3) == pytest.approx(0.7653606, abs=1e-6)

  assert ndcg_at_k(ranking, groups, 10) == approx(1.0)


# ---------------------------------------------------- properties -----------

def test_T10_ndcg_always_in_unit_interval():
  import random
  rng = random.Random(1234)
  pool = [f"d{i}" for i in range(40)]
  for _ in range(1000):
    n_groups = rng.randint(1, 5)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    members = shuffled[:rng.randint(n_groups, min(12, len(pool)))]
    groups = [set() for _ in range(n_groups)]
    for i, m in enumerate(members):
      groups[i % n_groups].add(m)
    ranking = pool[:]
    rng.shuffle(ranking)
    ranking = ranking[:rng.randint(0, len(pool))]
    for k in (1, 3, 5, 10):
      val = ndcg_at_k(ranking, groups, k)
      assert 0.0 <= val <= 1.0 + 1e-12, (val, groups, ranking[:12], k)


def test_T11_promoting_a_covering_doc_never_hurts():
  import random
  rng = random.Random(99)
  for _ in range(300):
    groups = [{"a"}, {"b"}]
    others = [f"x{i}" for i in range(10)]
    ranking = ["a"] + others[:4] + ["b"] + others[4:]
    rng.shuffle(others)
    before = score_question("q", "all_required", ranking, groups)
    # promote "b" one position earlier
    idx = ranking.index("b")
    promoted = ranking[:]
    promoted[idx - 1], promoted[idx] = promoted[idx], promoted[idx - 1]
    after = score_question("q", "all_required", promoted, groups)
    assert after.rr >= before.rr - 1e-12
    for k in (3, 5, 10):
      assert after.ndcg_at[k] >= before.ndcg_at[k] - 1e-12
      assert after.recall_at[k] >= before.recall_at[k] - 1e-12


# ------------------------------------------------------- edge cases --------

def test_T12_empty_ranking_scores_zero_without_exploding():
  groups = [{"a"}]
  assert coverage_ranks([], groups) == [INF]
  assert reciprocal_rank_coverage([], groups) == approx(0.0)
  assert group_recall_at_k([], groups, 5) == approx(0.0)
  assert sufficiency_at_k([], groups, 5) == approx(0.0)
  assert ndcg_at_k([], groups, 10) == approx(0.0)


def test_T13_duplicate_ids_in_ranking_dedupe_first_wins():
  groups = [{"b"}]
  assert coverage_ranks(["a", "a", "b"], groups) == [2]
  assert reciprocal_rank_coverage(["a", "a", "b"], groups) == approx(0.5)


def test_T14_overlapping_groups_rejected():
  with pytest.raises(ValueError, match="disjoint"):
    validate_groups([{"a", "b"}, {"b", "c"}])
  with pytest.raises(ValueError, match="disjoint"):
    ndcg_at_k(["a"], [{"a", "b"}, {"b", "c"}], 10)


def test_T15_empty_group_rejected():
  with pytest.raises(ValueError):
    validate_groups([set()])
  with pytest.raises(ValueError):
    validate_groups([])


def test_T16_rr_cap():
  groups = [{"a"}]
  ranking = [f"x{i}" for i in range(59)] + ["a"]     # "a" at rank 60
  assert reciprocal_rank_coverage(ranking, groups, cap=50) == approx(0.0)
  assert reciprocal_rank_coverage(ranking, groups, cap=None) == approx(1 / 60)


def test_T19_real_chunk_id_format_survives():
  groups = [{"2004.04906::p2::c0", "2004.04906::p2::c1"}]
  ranking = ["2404.16130::p0::c0", "2004.04906::p2::c1"]
  assert coverage_ranks(ranking, groups) == [2]
  assert reciprocal_rank_coverage(ranking, groups) == approx(0.5)


# ------------------------------------------------------- aggregation -------

def test_T17_macro_mean_and_strata():
  s1 = score_question("q1", "any_of", ["a"], [{"a"}])
  s2 = score_question("q2", "all_required", ["x"], [{"p"}, {"q"}])
  agg = aggregate([s1, s2])
  assert agg.mrr == approx(0.5)
  assert agg.n == 2

  strata = aggregate_by_stratum([s1, s2])
  assert strata["any_of"].mrr == approx(1.0)
  assert strata["all_required"].mrr == approx(0.0)
  assert strata["all"].mrr == approx(0.5)


def test_T18_macro_not_micro():
  """LOAD-BEARING. Questions are weighted equally, not by group count.

  q1: 1 group, covered.  q2: 3 groups, 1 covered.
  macro = mean(1.0, 0.333) = 0.6667.  micro = 2 covered / 4 total = 0.5.
  Micro would let a handful of multi-hop questions swing a 90-question headline.
  """
  q1 = score_question("q1", "any_of", ["a"], [{"a"}])
  q2 = score_question("q2", "all_required", ["p"], [{"p"}, {"q"}, {"r"}])
  agg = aggregate([q1, q2])
  assert agg.recall_at[5] == approx((1.0 + 1 / 3) / 2)
  assert agg.recall_at[5] == pytest.approx(0.6666666, abs=1e-6)
  assert agg.recall_at[5] != pytest.approx(0.5, abs=1e-6)


def test_paired_delta_ci_detects_a_real_improvement():
  baseline = [0.2] * 40
  candidate = [0.5] * 40
  mean_delta, lo, hi = paired_delta_ci(baseline, candidate, iters=2000, seed=7)
  assert mean_delta == approx(0.3)
  assert lo > 0.0 and hi > 0.0        # zero excluded -> a real effect


def test_paired_delta_ci_rejects_misaligned_input():
  with pytest.raises(ValueError, match="align"):
    paired_delta_ci([0.1, 0.2], [0.3])
