"""Each validation code exercised against a deliberately-broken golden set.

Built from inline dicts rather than files: a validator that only works on
well-formed input is untested, and these are the rules standing between a typo
and a silently depressed baseline.
"""

import pytest

from eval.golden_set import (
  GoldenSet,
  Severity,
  content_words,
  has_errors,
  question_chunk_jaccard,
  question_from_dict,
  validate_golden_set,
)

GOOD_ID = "2004.04906::p2::c0"
GOOD_ID2 = "2004.04906::p2::c1"
OTHER_ID = "2404.16130::p5::c0"


def make_set(questions, fingerprint="sha256:abc"):
  return GoldenSet(
    version="test",
    corpus_fingerprint=fingerprint,
    chunking={"chunk_size": 500, "chunk_overlap": 50},
    questions=tuple(question_from_dict(q) for q in questions),
  )


def q(qid="q1", semantics="any_of", groups=None, **kw):
  base = {
    "qid": qid,
    "question": "What is dense passage retrieval?",
    "semantics": semantics,
    "gold_groups": groups if groups is not None else [{"chunk_ids": [GOOD_ID]}],
  }
  base.update(kw)
  return base


def codes(issues, severity=None):
  return {i.code for i in issues if severity is None or i.severity is severity}


# ------------------------------------------------------------------ happy --

def test_valid_set_produces_no_issues():
  gs = make_set([q()])
  assert validate_golden_set(gs) == []
  assert not has_errors(validate_golden_set(gs))


def test_valid_multihop():
  gs = make_set([q(semantics="all_required",
                   groups=[{"chunk_ids": [GOOD_ID]}, {"chunk_ids": [OTHER_ID]}])])
  assert validate_golden_set(gs) == []


# ------------------------------------------------------------- structure ---

def test_DUP_QID():
  gs = make_set([q(qid="dupe"), q(qid="dupe")])
  assert "DUP_QID" in codes(validate_golden_set(gs), Severity.ERROR)


def test_EMPTY_GROUPS():
  gs = make_set([q(groups=[])])
  assert "EMPTY_GROUPS" in codes(validate_golden_set(gs), Severity.ERROR)


def test_EMPTY_GROUP():
  gs = make_set([q(semantics="all_required",
                   groups=[{"chunk_ids": [GOOD_ID]}, {"chunk_ids": []}])])
  assert "EMPTY_GROUP" in codes(validate_golden_set(gs), Severity.ERROR)


def test_BAD_ID_FORMAT():
  gs = make_set([q(groups=[{"chunk_ids": ["not-a-chunk-id"]}])])
  assert "BAD_ID_FORMAT" in codes(validate_golden_set(gs), Severity.ERROR)


def test_GROUPS_OVERLAP():
  """The one that would silently allow nDCG > 1."""
  gs = make_set([q(semantics="all_required",
                   groups=[{"chunk_ids": [GOOD_ID, GOOD_ID2]},
                           {"chunk_ids": [GOOD_ID2, OTHER_ID]}])])
  assert "GROUPS_OVERLAP" in codes(validate_golden_set(gs), Severity.ERROR)


def test_DUP_ID_IN_GROUP_is_a_warning_not_an_error():
  gs = make_set([q(groups=[{"chunk_ids": [GOOD_ID, GOOD_ID]}])])
  issues = validate_golden_set(gs)
  assert "DUP_ID_IN_GROUP" in codes(issues, Severity.WARN)
  assert not has_errors(issues)


# ------------------------------------------------------------- semantics ---

def test_SEMANTICS_MISMATCH_any_of_with_two_groups():
  gs = make_set([q(semantics="any_of",
                   groups=[{"chunk_ids": [GOOD_ID]}, {"chunk_ids": [OTHER_ID]}])])
  assert "SEMANTICS_MISMATCH" in codes(validate_golden_set(gs), Severity.ERROR)


def test_SEMANTICS_MISMATCH_all_required_with_one_group():
  gs = make_set([q(semantics="all_required", groups=[{"chunk_ids": [GOOD_ID]}])])
  assert "SEMANTICS_MISMATCH" in codes(validate_golden_set(gs), Severity.ERROR)


def test_SEMANTICS_MISMATCH_unknown_label():
  gs = make_set([q(semantics="sometimes")])
  assert "SEMANTICS_MISMATCH" in codes(validate_golden_set(gs), Severity.ERROR)


def test_SUSPICIOUS_MULTIHOP_adjacent_chunks_labelled_as_distinct_hops():
  """chunk_overlap=50 guarantees adjacent chunks share text, so two 'hops' one
  ordinal apart on the same page are almost certainly mislabelled alternatives."""
  gs = make_set([q(semantics="all_required",
                   groups=[{"chunk_ids": [GOOD_ID]}, {"chunk_ids": [GOOD_ID2]}])])
  issues = validate_golden_set(gs)
  assert "SUSPICIOUS_MULTIHOP" in codes(issues, Severity.WARN)
  assert not has_errors(issues)      # a warning, since a page can hold two real hops


def test_no_SUSPICIOUS_MULTIHOP_across_papers():
  gs = make_set([q(semantics="all_required",
                   groups=[{"chunk_ids": [GOOD_ID]}, {"chunk_ids": [OTHER_ID]}])])
  assert "SUSPICIOUS_MULTIHOP" not in codes(validate_golden_set(gs))


def test_N_GROUPS_EXCEEDS_K():
  groups = [{"chunk_ids": [f"2004.04906::p{i}::c0"]} for i in range(6)]
  gs = make_set([q(semantics="all_required", groups=groups)])
  assert "N_GROUPS_EXCEEDS_K" in codes(validate_golden_set(gs, reporting_ks=(5, 10)), Severity.WARN)


# --------------------------------------------------- collection-dependent --

def test_UNKNOWN_CHUNK_ID():
  gs = make_set([q()])
  issues = validate_golden_set(gs, known_chunk_ids={OTHER_ID})
  assert "UNKNOWN_CHUNK_ID" in codes(issues, Severity.ERROR)


def test_known_chunk_ids_none_skips_the_rule():
  gs = make_set([q()])
  assert "UNKNOWN_CHUNK_ID" not in codes(validate_golden_set(gs, known_chunk_ids=None))


def test_TEXT_DRIFT():
  gs = make_set([q(chunk_text_sha8={GOOD_ID: "aaaaaaaa"})])
  issues = validate_golden_set(gs, known_text_sha8={GOOD_ID: "bbbbbbbb"})
  assert "TEXT_DRIFT" in codes(issues, Severity.ERROR)


def test_no_TEXT_DRIFT_when_hashes_agree():
  gs = make_set([q(chunk_text_sha8={GOOD_ID: "aaaaaaaa"})])
  assert "TEXT_DRIFT" not in codes(validate_golden_set(gs, known_text_sha8={GOOD_ID: "aaaaaaaa"}))


def test_FINGERPRINT_MISMATCH():
  gs = make_set([q()], fingerprint="sha256:authored_against_this")
  issues = validate_golden_set(gs, collection_fingerprint="sha256:but_collection_is_this")
  assert "FINGERPRINT_MISMATCH" in codes(issues, Severity.ERROR)


# ---------------------------------------------------------------- leakage --

def test_jaccard_high_when_question_echoes_chunk_vocabulary():
  chunk = "Dense Passage Retrieval trains a bi-encoder with in-batch negatives on Natural Questions."
  echoing = "What does Dense Passage Retrieval train with in-batch negatives on Natural Questions?"
  paraphrased = "How are the two towers optimised during training, and on which benchmark?"
  assert question_chunk_jaccard(echoing, chunk) > question_chunk_jaccard(paraphrased, chunk)
  assert question_chunk_jaccard(echoing, chunk) > 0.35


def test_content_words_strips_stopwords_and_short_tokens():
  assert content_words("What is the RAG model?") == {"rag", "model"}


def test_jaccard_handles_empty_input():
  assert question_chunk_jaccard("", "text") == 0.0
  assert question_chunk_jaccard("the and of", "text") == 0.0


# ------------------------------------------------------------ round-trip --

def test_group_sets_is_what_metrics_consumes():
  question = question_from_dict(q(semantics="all_required",
                                  groups=[{"chunk_ids": [GOOD_ID, GOOD_ID2]},
                                          {"chunk_ids": [OTHER_ID]}]))
  assert question.group_sets == (frozenset({GOOD_ID, GOOD_ID2}), frozenset({OTHER_ID}))
  assert question.n_groups == 2
  assert set(question.all_chunk_ids) == {GOOD_ID, GOOD_ID2, OTHER_ID}
