"""Reciprocal Rank Fusion: the arithmetic, and the properties it must keep.

The fusion step is where a ranking bug is hardest to notice -- both arms keep
returning sensible documents, and the fused list still looks like a ranking. So
the worked example below is checked against hand-computed values rather than
against whatever the implementation currently produces.
"""

import pytest

from core.hybrid_retriever import DENSE_ARM, RRF_K, SPARSE_ARM, rrf_fuse


def order(fused):
  return [chunk_id for chunk_id, _score, _ranks in fused]


def scores(fused):
  return {chunk_id: score for chunk_id, score, _ranks in fused}


DENSE = ["a", "b", "c"]
SPARSE = ["b", "d", "a"]


def test_worked_example_matches_hand_computed_rrf():
  fused = rrf_fuse({DENSE_ARM: DENSE, SPARSE_ARM: SPARSE}, k=60)
  got = scores(fused)

  assert got["a"] == pytest.approx(1 / 61 + 1 / 63)   # dense 1, sparse 3
  assert got["b"] == pytest.approx(1 / 62 + 1 / 61)   # dense 2, sparse 1
  assert got["c"] == pytest.approx(1 / 63)            # dense 3 only
  assert got["d"] == pytest.approx(1 / 62)            # sparse 2 only

  # The point of RRF in one assertion: b, which neither arm ranked first in
  # both lists, beats a, which one arm ranked first -- agreement across arms
  # outweighs a single arm's confidence.
  assert order(fused) == ["b", "a", "d", "c"]


def test_default_k_is_the_published_constant():
  assert RRF_K == 60


def test_agreement_beats_a_single_strong_arm():
  # Found at rank 3 by both arms vs rank 1 by one arm only.
  fused = rrf_fuse({DENSE_ARM: ["solo", "x", "agreed"], SPARSE_ARM: ["y", "z", "agreed"]}, k=60)
  assert order(fused)[0] == "agreed"


def test_single_list_fusion_preserves_that_list_exactly():
  fused = rrf_fuse({DENSE_ARM: ["x", "y", "z"]})
  assert order(fused) == ["x", "y", "z"]


def test_zero_weight_reduces_to_the_other_arm():
  fused = rrf_fuse({DENSE_ARM: DENSE, SPARSE_ARM: SPARSE}, k=60,
                   weights={DENSE_ARM: 1.0, SPARSE_ARM: 0.0})
  # Ids seen only by the muted arm still appear (at score 0) but can never
  # outrank a scoring document.
  assert order(fused)[:3] == DENSE


def test_weights_are_applied_not_ignored():
  balanced = scores(rrf_fuse({DENSE_ARM: ["a"], SPARSE_ARM: ["b"]}, k=60))
  tilted = scores(rrf_fuse({DENSE_ARM: ["a"], SPARSE_ARM: ["b"]}, k=60,
                           weights={DENSE_ARM: 2.0, SPARSE_ARM: 1.0}))
  assert balanced["a"] == pytest.approx(balanced["b"])
  assert tilted["a"] == pytest.approx(2 * tilted["b"])


def test_fusion_never_invents_an_id():
  fused = rrf_fuse({DENSE_ARM: DENSE, SPARSE_ARM: SPARSE})
  assert set(order(fused)) == set(DENSE) | set(SPARSE)


def test_per_arm_ranks_are_recorded_with_none_for_absence():
  fused = dict((cid, ranks) for cid, _score, ranks in
               rrf_fuse({DENSE_ARM: DENSE, SPARSE_ARM: SPARSE}))
  assert fused["a"] == {DENSE_ARM: 1, SPARSE_ARM: 3}
  assert fused["c"] == {DENSE_ARM: 3, SPARSE_ARM: None}
  assert fused["d"] == {DENSE_ARM: None, SPARSE_ARM: 2}


def test_duplicate_id_within_one_arm_is_not_double_counted():
  once = scores(rrf_fuse({DENSE_ARM: ["a", "b"]}))
  twice = scores(rrf_fuse({DENSE_ARM: ["a", "a", "b"]}))
  assert twice["a"] == pytest.approx(once["a"])
  assert twice["b"] == pytest.approx(once["b"])   # and b keeps rank 2, not 3


def test_ties_break_on_id_deterministically():
  # Both appear at rank 1 of exactly one arm -> identical scores.
  fused = rrf_fuse({DENSE_ARM: ["zzz"], SPARSE_ARM: ["aaa"]})
  assert order(fused) == ["aaa", "zzz"]


def test_arm_order_does_not_affect_the_result():
  forward = rrf_fuse({DENSE_ARM: DENSE, SPARSE_ARM: SPARSE})
  backward = rrf_fuse({SPARSE_ARM: SPARSE, DENSE_ARM: DENSE})
  assert order(forward) == order(backward)
  assert scores(forward) == pytest.approx(scores(backward))


def test_larger_k_flattens_the_rank_discount():
  # Higher k means later ranks are discounted less steeply -- the whole content
  # of the sensitivity sweep, asserted so the knob is known to do something.
  tight = scores(rrf_fuse({DENSE_ARM: ["a", "b"]}, k=1))
  loose = scores(rrf_fuse({DENSE_ARM: ["a", "b"]}, k=1000))
  assert tight["a"] / tight["b"] > loose["a"] / loose["b"]


def test_empty_input_is_empty_output():
  assert rrf_fuse({DENSE_ARM: [], SPARSE_ARM: []}) == []


def test_non_positive_k_is_rejected():
  with pytest.raises(ValueError):
    rrf_fuse({DENSE_ARM: ["a"]}, k=0)
