"""The sparse arm: tokenization decisions and BM25 scoring properties.

These pin the tokenization scheme as *behaviour*, not as a comment. It is the
one part of BM25 that silently changes what "matches" means, and this corpus
has two specific hazards -- PDF ligatures (in ~10% of chunks) and hyphenated
technical compounds -- that a naive tokenizer mangles while still producing
plausible rankings.

Pure stdlib, no vectorstore, same constraint as the metrics tests.
"""

import math

import pytest

from core.bm25 import (
  DEFAULT_B,
  DEFAULT_K1,
  OKAPI_EPSILON,
  TOKENIZATION_VERSION,
  BM25Index,
  build_index,
  index_stats,
  tokenize,
)


# ---------------------------------------------------------------- tokenizer --

def test_tokenization_version_is_declared():
  # Its absence is what would let a tokenizer change slip through unrecorded.
  assert TOKENIZATION_VERSION


def test_ligatures_are_normalized():
  # PyPDF hands back the real ligature glyph; without NFKD this splits into
  # "identi" + "cation" and the term is simply lost.
  assert "identification" in tokenize("identiﬁcation")
  assert "workflow" in tokenize("workﬂow")


def test_accents_fold_to_base_letters():
  assert tokenize("Càrdenas") == tokenize("Cardenas")


def test_casefolded():
  assert tokenize("ColBERTv2") == ["colbertv2"]


def test_compound_emits_whole_and_parts():
  assert tokenize("retrieval-augmented") == ["retrieval-augmented", "retrieval", "augmented"]


def test_compound_lets_spaced_and_hyphenated_forms_meet():
  hyphenated = set(tokenize("retrieval-augmented generation"))
  spaced = set(tokenize("retrieval augmented generation"))
  assert spaced <= hyphenated          # the spaced query's terms are all present
  assert "retrieval-augmented" in hyphenated   # and the exact form still scores extra


def test_arxiv_ids_and_versioned_names_survive():
  assert tokenize("2004.04906") == ["2004.04906", "2004", "04906"]
  assert tokenize("gpt-oss-120b")[0] == "gpt-oss-120b"


def test_single_character_tokens_are_kept():
  # "k" in "top-k" is a real query term in this domain.
  assert "k" in tokenize("top-k retrieval")


def test_no_stemming():
  # Deliberate: BM25 is here for exact terms. If this ever starts passing as
  # equal, the tokenizer grew a stemmer and every hybrid number moved with it.
  assert tokenize("retriever") != tokenize("retrieval")


def test_no_stopword_removal():
  assert "the" in tokenize("the retriever")


def test_punctuation_and_empty_input():
  assert tokenize("") == []
  assert tokenize("--- ,. ;") == []


# -------------------------------------------------------------------- index --

CORPUS = [
  ("d1", "dense retrieval with a bi-encoder over passages"),
  ("d2", "dense retrieval and sparse retrieval fused with reciprocal rank fusion"),
  ("d3", "colbertv2 uses residual compression for late interaction"),
  ("d4", "retrieval retrieval retrieval retrieval"),
]


def idx(**kwargs):
  return build_index(CORPUS, **kwargs)


def test_index_shape():
  index = idx()
  assert index.n_docs == 4
  assert index.vocabulary_size == len({t for _, text in CORPUS for t in tokenize(text)})
  assert index.avgdl == pytest.approx(sum(len(tokenize(t)) for _, t in CORPUS) / 4)


def test_rejects_misaligned_inputs():
  with pytest.raises(ValueError):
    BM25Index(["a", "b"], ["only one"])


def test_rejects_duplicate_ids():
  with pytest.raises(ValueError):
    BM25Index(["a", "a"], ["x", "y"])


def test_rejects_unknown_idf_variant():
  with pytest.raises(ValueError):
    BM25Index(["a"], ["x"], idf_variant="bm25f")


def test_empty_corpus_does_not_explode():
  index = BM25Index([], [])
  assert index.search("anything", 5) == []


# ------------------------------------------------------------------ scoring --

def test_rare_term_beats_common_term():
  index = idx()
  # "colbertv2" appears in one doc; "retrieval" in three.
  assert index.idf("colbertv2") > index.idf("retrieval")


def test_idf_is_monotone_decreasing_in_document_frequency():
  index = idx()
  terms = sorted(index.postings, key=index.document_frequency)
  idfs = [index.idf(t) for t in terms]
  assert idfs == sorted(idfs, reverse=True)


def test_search_returns_the_document_containing_the_query_term():
  index = idx()
  assert index.search("residual compression", 1)[0][0] == "d3"


def test_unknown_term_scores_nothing():
  index = idx()
  assert index.search("neo4j cypher", 5) == []


def test_term_frequency_saturates():
  # d4 has 4x "retrieval" but must not score 4x a single occurrence: that is what
  # k1 is for, and it is the reason BM25 is not just weighted term counting.
  index = BM25Index(["once", "many"], ["retrieval", "retrieval " * 8])
  scores = dict(index.search("retrieval", 2))
  assert scores["many"] < 8 * scores["once"]


def test_shorter_document_wins_at_equal_term_frequency():
  # Length normalisation (b): one hit in a short passage is stronger evidence
  # than one hit buried in a long one.
  index = BM25Index(["short", "long"],
                    ["reranking matters", "reranking " + "filler words here " * 30])
  ranked = index.search("reranking", 2)
  assert ranked[0][0] == "short"


def test_repeated_query_term_counts_twice():
  index = idx()
  once = dict(index.search("colbertv2", 4))["d3"]
  twice = dict(index.search("colbertv2 colbertv2", 4))["d3"]
  assert twice == pytest.approx(2 * once)


def test_ties_break_on_chunk_id_not_insertion_order():
  # Two identical documents score identically; the order must not depend on
  # which happened to be inserted first, or rankings wobble between runs.
  forward = BM25Index(["b_id", "a_id"], ["reranking", "reranking"])
  backward = BM25Index(["a_id", "b_id"], ["reranking", "reranking"])
  assert [cid for cid, _ in forward.search("reranking", 2)] == ["a_id", "b_id"]
  assert [cid for cid, _ in backward.search("reranking", 2)] == ["a_id", "b_id"]


def test_k_truncates():
  index = idx()
  assert len(index.search("retrieval", 2)) == 2


def test_defaults_are_the_documented_constants():
  index = idx()
  assert (index.k1, index.b) == (DEFAULT_K1, DEFAULT_B)
  assert index.idf_variant == "lucene"


# ------------------------------------------------------------------ explain --

def test_explain_contributions_sum_to_the_score():
  index = idx()
  query = "dense retrieval fusion"
  score = dict(index.search(query, 4))["d2"]
  assert sum(v for _, v in index.explain(query, "d2")) == pytest.approx(score)


def test_explain_rejects_unknown_chunk():
  with pytest.raises(KeyError):
    idx().explain("dense", "nope")


# ------------------------------------------------------- idf variant switch --

def test_lucene_idf_is_never_negative():
  # A term in every document is merely cheap under the shipped variant, not a
  # penalty that can drag a genuinely matching document below a non-matching one.
  index = BM25Index(["a", "b"], ["shared term", "shared term"])
  assert index.idf("shared") > 0


def test_okapi_idf_floors_negative_values_like_rank_bm25():
  index = BM25Index(["a", "b"], ["shared term", "shared other"], idf_variant="okapi")
  raw = math.log(2 - 2 + 0.5) - math.log(2 + 0.5)     # df=2 of N=2 -> negative
  assert raw < 0
  assert index.idf("shared") == pytest.approx(OKAPI_EPSILON * _mean_idf(index))


def _mean_idf(index):
  # rank_bm25 averages over the PRE-floor idfs, which is what we reproduce.
  total = 0.0
  for term, posting in index.postings.items():
    df = len(posting)
    total += math.log(index.n_docs - df + 0.5) - math.log(df + 0.5)
  return total / len(index.postings)


def test_index_stats_records_what_would_invalidate_a_comparison():
  stats = index_stats(idx())
  assert stats["tokenization_version"] == TOKENIZATION_VERSION
  assert stats["idf_variant"] == "lucene"
  assert stats["n_docs"] == 4
