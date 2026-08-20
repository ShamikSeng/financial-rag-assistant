"""BM25 Okapi over the frozen corpus -- hand-rolled, pure stdlib.

Why hand-rolled
---------------
There is no performance case for a library here: the frozen corpus is 2131
chunks / 459K tokens / 37.5K vocabulary, which indexes in ~0.7s including the
Chroma read. So the only thing a dependency buys is opacity, in a project whose
eval scorer (`eval/metrics.py`) is deliberately pure stdlib for exactly the same
reason, and which already rejected RAGAS partly for being "somewhat black-box".

Correctness is not taken on trust either --
`scripts/verify_bm25_against_rank_bm25.py` checks this implementation against
`rank_bm25` (temporarily installed, deliberately NOT a dependency). That is what
`idf_variant="okapi"` below exists for: it makes the cross-check an exact
equality assertion rather than an eyeball.

This module imports nothing outside the standard library -- no LangChain, no
Chroma, no numpy -- so its tests run without a vector store, in the same spirit
as the metrics purity constraint. Everything that touches the collection lives
in `hybrid_retriever.py`.

Tokenization is a first-class decision here, not an implementation detail --
see `tokenize()`.
"""

import math
import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Bump on ANY change to tokenize(). A BM25 index built under a different
# tokenization is not comparable to one built under this one, and a silently
# re-tokenized sparse arm would shift every hybrid number without shifting
# anything visible -- the same failure shape CHUNKING_VERSION guards against in
# chunk_ids.py.
TOKENIZATION_VERSION = "v1"

# Runs of alphanumerics, optionally joined by internal punctuation. The joiners
# are what keep "top-k", "gfm-rag" and "2004.04906" as single terms.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")
_JOINER_RE = re.compile(r"[._\-/]")

# Robertson/Lucene defaults. k1 controls term-frequency saturation, b controls
# length normalization. Not tuned -- see the RRF constant decision in
# PROJECT_LOG.md for why nothing in this phase is fitted to the golden set.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

# rank_bm25.BM25Okapi's own defaults, used only by the equivalence check.
OKAPI_K1 = 1.5
OKAPI_EPSILON = 0.25


def tokenize(text: str) -> List[str]:
  """Text -> BM25 terms. The scheme is deliberate; see PROJECT_LOG.md.

  1. NFKD normalize, then drop combining marks. Measured on this corpus: 212 of
     2131 chunks (~10%) contain PDF ligature characters, and 1663 contain some
     non-ASCII. Without this step an ASCII tokenizer splits "identification"
     (written with an fi-ligature by the PDF) into "identi" + "cation" across a
     tenth of the corpus. It also folds accented author names onto their base
     letters so either spelling matches.
  2. Casefold, so matching is case-insensitive.
  3. Emit compounds AND their parts. Punctuation-joined compounds are everywhere
     here -- retrieval-augmented (x380), multi-hop (x261), end-to-end (x152),
     top-k (x94), re-ranking (x92), plus method names like gfm-rag. Indexing both
     forms means a query writing "retrieval augmented generation" still matches a
     chunk writing "retrieval-augmented generation", while the intact compound
     stays a high-IDF term that rewards the exact form.

  Explicitly NOT done:

  * No stemming. BM25 is in this pipeline *because* it catches exact terms a
    dense bi-encoder blurs past (ColBERTv2 vs ColBERT, nDCG@10 vs "a ranking
    metric"). A stemmer spends precisely that advantage, and morphological
    variation is the dense arm's job in the fusion. Plural folding is a
    candidate future *measured* change, not a silent one.
  * No stopword removal. IDF already drives near-universal terms to ~0 weight,
    so a list is a redundant knob that can only break exact phrases.
    (`eval/golden_set.py` has a stopword list for its Jaccard leakage check --
    unrelated, deliberately not reused.)
  * No minimum token length and no number stripping. The "k" in "top-k", arXiv
    ids and benchmark numbers all carry signal; IDF discounts the rest.
  """
  normalized = unicodedata.normalize("NFKD", text).casefold()
  normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

  tokens: List[str] = []
  for match in _TOKEN_RE.finditer(normalized):
    token = match.group(0)
    tokens.append(token)
    if _JOINER_RE.search(token):
      tokens.extend(part for part in _JOINER_RE.split(token) if part)
  return tokens


class BM25Index:
  """In-memory BM25 over (chunk_id, text) pairs.

  Scoring is the standard Okapi form:

      score(q, d) = SUM_t idf(t) * tf(t,d) * (k1 + 1)
                    / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

  `idf_variant` selects how idf(t) is computed:

  * "lucene" (shipped default) -- ln(1 + (N - df + 0.5)/(df + 0.5)), which is
    strictly positive, so a term appearing in more than half the corpus is
    merely cheap rather than actively penalising.
  * "okapi" -- ln(N - df + 0.5) - ln(df + 0.5), which goes negative for such
    terms, with rank_bm25's epsilon floor (epsilon * mean_idf) applied to those.
    Present only so the reference cross-check can assert exact score equality
    against rank_bm25 rather than approximate agreement.
  """

  def __init__(self, ids: Sequence[str], texts: Sequence[str],
               k1: float = DEFAULT_K1, b: float = DEFAULT_B,
               idf_variant: str = "lucene", epsilon: float = OKAPI_EPSILON):
    if len(ids) != len(texts):
      raise ValueError(f"ids and texts must align: {len(ids)} vs {len(texts)}")
    if idf_variant not in ("lucene", "okapi"):
      raise ValueError(f"idf_variant must be 'lucene' or 'okapi', got {idf_variant!r}")
    if len(set(ids)) != len(ids):
      raise ValueError("duplicate chunk ids in BM25Index -- ids must be unique")

    self.k1 = k1
    self.b = b
    self.idf_variant = idf_variant
    self.epsilon = epsilon
    self.tokenization_version = TOKENIZATION_VERSION

    self.ids: List[str] = list(ids)
    self._position: Dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}
    self.doc_len: List[int] = []
    self.postings: Dict[str, Dict[int, int]] = {}

    for doc_index, text in enumerate(texts):
      terms = tokenize(text)
      self.doc_len.append(len(terms))
      counts: Dict[str, int] = {}
      for term in terms:
        counts[term] = counts.get(term, 0) + 1
      for term, freq in counts.items():
        self.postings.setdefault(term, {})[doc_index] = freq

    self.n_docs = len(self.ids)
    self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
    self._idf: Dict[str, float] = self._compute_idf()

  # -- construction helpers ------------------------------------------------

  def _compute_idf(self) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    if not self.n_docs:
      return idf

    if self.idf_variant == "lucene":
      for term, posting in self.postings.items():
        df = len(posting)
        idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
      return idf

    # okapi: rank_bm25.BM25Okapi, including its negative-idf epsilon floor
    negative = []
    total = 0.0
    for term, posting in self.postings.items():
      df = len(posting)
      value = math.log(self.n_docs - df + 0.5) - math.log(df + 0.5)
      idf[term] = value
      total += value
      if value < 0:
        negative.append(term)
    floor = self.epsilon * (total / len(idf))
    for term in negative:
      idf[term] = floor
    return idf

  # -- introspection -------------------------------------------------------

  @property
  def vocabulary_size(self) -> int:
    return len(self.postings)

  def document_frequency(self, term: str) -> int:
    return len(self.postings.get(term, {}))

  def idf(self, term: str) -> float:
    return self._idf.get(term, 0.0)

  # -- scoring -------------------------------------------------------------

  def _term_score(self, term: str, doc_index: int) -> float:
    posting = self.postings.get(term)
    if not posting:
      return 0.0
    tf = posting.get(doc_index)
    if not tf:
      return 0.0
    norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
    return self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)

  def score_all(self, query: str) -> Dict[int, float]:
    """doc_index -> score, for documents matching at least one query term.

    Iterates the query's token LIST, not its set: a term repeated in the query
    contributes twice, which is both standard BM25 and what rank_bm25 does.
    """
    scores: Dict[int, float] = {}
    for term in tokenize(query):
      posting = self.postings.get(term)
      if not posting:
        continue
      for doc_index in posting:
        scores[doc_index] = scores.get(doc_index, 0.0) + self._term_score(term, doc_index)
    return scores

  def search(self, query: str, k: int) -> List[Tuple[str, float]]:
    """Top-k (chunk_id, score), best first.

    Tie-break is (-score, chunk_id) rather than insertion order. Adjacent chunks
    overlap by 50 tokens, so exact score ties are common; leaving them to dict
    ordering would let the ranking wobble between runs and manufacture fake
    deltas between phases -- the same reason DenseChromaRetriever sorts its ties.
    """
    scores = self.score_all(query)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], self.ids[item[0]]))
    return [(self.ids[doc_index], score) for doc_index, score in ranked[:k]]

  def explain(self, query: str, chunk_id: str) -> List[Tuple[str, float]]:
    """Per-term contributions of `query` to `chunk_id`, largest first.

    The answer to "how do you know BM25 matched what you think it matched" --
    and the fastest way to tell a real lexical hit from one carried by a single
    junk token that PDF extraction glued together.
    """
    doc_index = self._position.get(chunk_id)
    if doc_index is None:
      raise KeyError(f"unknown chunk_id: {chunk_id!r}")
    contributions: Dict[str, float] = {}
    for term in tokenize(query):
      value = self._term_score(term, doc_index)
      if value:
        contributions[term] = contributions.get(term, 0.0) + value
    return sorted(contributions.items(), key=lambda item: (-item[1], item[0]))


def build_index(pairs: Iterable[Tuple[str, str]], **kwargs) -> BM25Index:
  """Convenience: build from an iterable of (chunk_id, text)."""
  items = list(pairs)
  return BM25Index([cid for cid, _ in items], [text for _, text in items], **kwargs)


def index_stats(index: BM25Index) -> Mapping[str, object]:
  """Recorded into run_meta so a sparse-arm change is visible after the fact."""
  return {
    "n_docs": index.n_docs,
    "vocabulary_size": index.vocabulary_size,
    "avgdl": round(index.avgdl, 3),
    "k1": index.k1,
    "b": index.b,
    "idf_variant": index.idf_variant,
    "tokenization_version": index.tokenization_version,
  }
