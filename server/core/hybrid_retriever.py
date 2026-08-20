"""Phase 2: hybrid retrieval -- dense (Chroma) + sparse (BM25), fused with RRF.

Why this exists
---------------
The Phase 1 baseline is naive top-k dense similarity, and its structural
weakness on this corpus is exact-term matching: a bi-encoder blurs ColBERTv2
toward ColBERT, GFM-RAG toward "graph RAG", nDCG@10 toward "a ranking metric".
BM25 does not blur those, and dense handles the paraphrases BM25 cannot. Fusing
the two ranked lists is the cheapest way to get both.

Fusion is Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):

    RRF(d) = SUM_arms w_arm / (k + rank_arm(d))

`k = 60` is the constant from that paper, chosen a priori. It is deliberately
NOT tuned against the golden set -- sweeping it on the same 89 questions the
headline number is claimed from would be test-set leakage with no held-out data
to confirm it generalises. A sweep is run separately and reported only as a
labelled robustness diagnostic. Weights default to 1.0/1.0 (vanilla RRF); the
weighted form exists for that exploratory table and is not what ships.

RRF works on *ranks*, never on raw scores, which is exactly why it suits this
pipeline: Chroma returns squared L2 distances (lower is better, unnormalised)
and BM25 returns unbounded relevance scores (higher is better). There is no
principled way to put those on one scale, and every attempt to (min-max, z-score)
smuggles in a per-query normalisation that shifts with the score distribution.
Ranks sidestep the whole problem.

The BM25 index is built FROM THE CHROMA COLLECTION rather than by re-parsing the
PDFs, so both arms provably index the same text under the same chunk ids. A
separately-built sparse index would be free to drift from the dense one on any
future re-chunk, and the drift would be invisible -- both arms would still
return plausible results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from core.bm25 import BM25Index, index_stats
from core.chunk_ids import text_sha8
from utils.logger import logger

# Cormack et al. 2009. Chosen a priori, not tuned -- see module docstring.
RRF_K = 60

# Per-arm candidate depth. Fixed rather than equal to the caller's k so that
# /chat asking for 3 documents still fuses over 50 + 50 real candidates instead
# of degenerating into "the top 3 of each list". It is also the shape Phase 3
# needs (retrieve wide, rerank down).
CANDIDATE_POOL = 50

DENSE_ARM = "dense"
SPARSE_ARM = "bm25"


@dataclass(frozen=True)
class FusedHit:
  chunk_id: str
  rank: int
  rrf_score: float
  ranks: Mapping[str, Optional[int]]      # arm -> rank in that arm (None = absent)
  l2_distance: Optional[float] = None     # dense arm only
  bm25_score: Optional[float] = None      # sparse arm only
  page_content: str = ""
  metadata: Mapping[str, object] = field(default_factory=dict)

  @property
  def found_by(self) -> str:
    dense = self.ranks.get(DENSE_ARM) is not None
    sparse = self.ranks.get(SPARSE_ARM) is not None
    if dense and sparse:
      return "both"
    if dense:
      return DENSE_ARM
    if sparse:
      return SPARSE_ARM
    return "neither"


def rrf_fuse(ranked_lists: Mapping[str, Sequence[str]], k: int = RRF_K,
             weights: Optional[Mapping[str, float]] = None) -> List[Tuple[str, float, Dict[str, Optional[int]]]]:
  """Fuse named ranked lists of ids. Returns (id, score, per-arm ranks), best first.

  Ties are broken by (-score, id). Ties are not rare here: two ids that each
  appear at the same rank in exactly one arm score identically, and adjacent
  chunks overlap by 50 tokens. Left to dict ordering the ranking would wobble
  between runs, which is how phases acquire fake +/-0.01 "improvements".
  """
  if k <= 0:
    raise ValueError(f"RRF k must be positive, got {k}")

  arms = list(ranked_lists.keys())
  weights = dict(weights or {})
  for arm in arms:
    weights.setdefault(arm, 1.0)

  scores: Dict[str, float] = {}
  ranks: Dict[str, Dict[str, Optional[int]]] = {}

  for arm in arms:
    seen = set()
    rank = 0
    for chunk_id in ranked_lists[arm]:
      if chunk_id in seen:      # a malformed arm must not double-count
        continue
      seen.add(chunk_id)
      rank += 1
      scores[chunk_id] = scores.get(chunk_id, 0.0) + weights[arm] / (k + rank)
      ranks.setdefault(chunk_id, {a: None for a in arms})[arm] = rank

  ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
  return [(chunk_id, score, ranks[chunk_id]) for chunk_id, score in ordered]


class HybridRetriever:
  """Dense + BM25 + RRF over one provider's Chroma collection.

  Built once and cached per provider (`get_hybrid_retriever`): the BM25 index
  costs a full collection read plus a tokenisation pass (~0.7s for 2131 chunks),
  which is cheap once and wasteful per request.
  """

  name = "hybrid_rrf"

  def __init__(self, provider: str = "groq", rrf_k: int = RRF_K,
               pool: int = CANDIDATE_POOL,
               weights: Optional[Mapping[str, float]] = None,
               stable_ties: bool = True):
    from core.vector_database import load_vectorstore   # local: avoids an import cycle

    self.provider = provider
    self.rrf_k = rrf_k
    self.pool = pool
    self.weights = dict(weights or {DENSE_ARM: 1.0, SPARSE_ARM: 1.0})
    self.stable_ties = stable_ties

    self._vs = load_vectorstore(provider)
    self._index, self._payload = self._build_sparse_index()

  # -- construction --------------------------------------------------------

  def _build_sparse_index(self) -> Tuple[BM25Index, Dict[str, Tuple[str, dict]]]:
    got = self._vs._collection.get(include=["documents", "metadatas"])
    keys: List[str] = []
    texts: List[str] = []
    payload: Dict[str, Tuple[str, dict]] = {}

    for chroma_id, document, metadata in zip(got["ids"], got["documents"], got["metadatas"]):
      metadata = dict(metadata or {})
      key = join_key(metadata, document, chroma_id)
      keys.append(key)
      texts.append(document)
      payload[key] = (document, metadata)

    index = BM25Index(keys, texts)
    logger.info(
      f"[hybrid:{self.provider}] BM25 index built: {index.n_docs} docs, "
      f"{index.vocabulary_size} terms, avgdl={index.avgdl:.1f}"
    )
    return index, payload

  # -- arms ----------------------------------------------------------------

  def dense_arm(self, query: str, depth: int) -> List[Tuple[str, float]]:
    """(key, l2_distance) in rank order, with the SAME stable tie-break the
    Phase 1 dense baseline uses -- otherwise the dense arm inside the hybrid is
    not the system the hybrid is being compared against."""
    scored = self._vs.similarity_search_with_score(query, k=depth)
    if self.stable_ties:
      scored = sorted(scored, key=lambda ds: (ds[1], ds[0].metadata.get("chunk_id") or ""))
    out = []
    for doc, distance in scored:
      metadata = dict(doc.metadata or {})
      out.append((join_key(metadata, doc.page_content, None), float(distance)))
      self._payload.setdefault(out[-1][0], (doc.page_content, metadata))
    return out

  def sparse_arm(self, query: str, depth: int) -> List[Tuple[str, float]]:
    return self._index.search(query, depth)

  # -- fusion --------------------------------------------------------------

  def retrieve(self, query: str, k: int) -> List[FusedHit]:
    depth = max(k, self.pool)
    dense = self.dense_arm(query, depth)
    sparse = self.sparse_arm(query, depth)

    dense_distance = dict(dense)
    sparse_score = dict(sparse)

    fused = rrf_fuse(
      {DENSE_ARM: [cid for cid, _ in dense], SPARSE_ARM: [cid for cid, _ in sparse]},
      k=self.rrf_k,
      weights=self.weights,
    )

    hits: List[FusedHit] = []
    for rank, (chunk_id, score, ranks) in enumerate(fused[:k], start=1):
      page_content, metadata = self._payload.get(chunk_id, ("", {}))
      hits.append(FusedHit(
        chunk_id=chunk_id,
        rank=rank,
        rrf_score=score,
        ranks=ranks,
        l2_distance=dense_distance.get(chunk_id),
        bm25_score=sparse_score.get(chunk_id),
        page_content=page_content,
        metadata=metadata,
      ))
    return hits

  def as_documents(self, query: str, k: int):
    from langchain_core.documents import Document
    return [Document(page_content=h.page_content, metadata=dict(h.metadata))
            for h in self.retrieve(query, k)]

  def explain(self, query: str, chunk_id: str):
    """Per-term BM25 contributions -- see BM25Index.explain."""
    return self._index.explain(query, chunk_id)

  @property
  def stats(self) -> Mapping[str, object]:
    return {
      **index_stats(self._index),
      "rrf_k": self.rrf_k,
      "candidate_pool": self.pool,
      "weights": dict(self.weights),
    }


def join_key(metadata: Mapping[str, object], text: str, chroma_id: Optional[str]) -> str:
  """The id both arms fuse on.

  Normally `metadata["chunk_id"]` -- the deterministic Phase 1 id, and the only
  identity that survives a langchain_community retrieval round-trip. Chunks
  added through the ad-hoc PDF upload endpoint predate that scheme and have no
  chunk_id, and the dense arm cannot see Chroma's primary key at all, so those
  fall back to a content-derived key computed identically on both sides. Without
  the fallback such a chunk would appear as two different documents to the
  fusion and be double-counted.
  """
  chunk_id = metadata.get("chunk_id")
  if chunk_id:
    return str(chunk_id)
  return f"text:{text_sha8(text)}"



# ---------------------------------------------------------------------------
# process-wide cache
# ---------------------------------------------------------------------------

_CACHE: Dict[Tuple[str, int, int], HybridRetriever] = {}


def get_hybrid_retriever(provider: str = "groq", rrf_k: int = RRF_K,
                         pool: int = CANDIDATE_POOL) -> HybridRetriever:
  key = (provider, rrf_k, pool)
  if key not in _CACHE:
    _CACHE[key] = HybridRetriever(provider, rrf_k=rrf_k, pool=pool)
  return _CACHE[key]


def reset_hybrid_cache(provider: Optional[str] = None) -> None:
  """Drop cached indexes. MUST be called after anything writes to a collection.

  Without this an upload leaves the sparse arm frozen at the pre-upload corpus
  while the dense arm sees the new chunks -- a hybrid that silently retrieves
  from two different corpora and still returns confident results.
  """
  if provider is None:
    _CACHE.clear()
    return
  for key in [k for k in _CACHE if k[0] == provider]:
    del _CACHE[key]


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------

def build_langchain_retriever(provider: str, k: int):
  """A langchain_core Retriever wrapping the hybrid path.

  Kept as a factory returning a class instance rather than a module-level class
  definition so that importing this module never requires langchain_core -- the
  eval harness and the tests use the plain `HybridRetriever` above.
  """
  from langchain_core.retrievers import BaseRetriever

  class _HybridLangChainRetriever(BaseRetriever):
    provider: str
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager=None):
      return get_hybrid_retriever(self.provider).as_documents(query, self.k)

  return _HybridLangChainRetriever(provider=provider, k=k)
