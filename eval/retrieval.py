"""The retriever-under-test interface, and the Phase 1 dense baseline.

The interface is deliberately narrow -- `retrieve(query, k) -> ranked chunks` --
so Phase 2 (hybrid BM25 + dense + RRF) and Phase 3 (BGE cross-encoder rerank)
add new classes here and change ZERO lines of metrics.py. That is the point of
building the harness before those phases rather than alongside them.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from eval._bootstrap import bootstrap_server_context

bootstrap_server_context()

from core.chunk_ids import CHUNK_ID_RE  # noqa: E402
from core.vector_database import load_vectorstore  # noqa: E402


@dataclass(frozen=True)
class RetrievedChunk:
  chunk_id: Optional[str]
  rank: int
  page_content: str
  metadata: Mapping[str, Any]
  # Phase 2: a hit can now come from a retriever with no notion of distance. A
  # BM25-only hit has no l2_distance and a dense-only hit has no bm25_score, so
  # both are optional rather than being faked with a sentinel that later reads
  # as a real measurement.
  l2_distance: Optional[float] = None
  score: Optional[float] = None                       # fusion score, higher is better
  ranks: Mapping[str, Optional[int]] = field(default_factory=dict)   # arm -> rank


class RetrieverUnderTest(Protocol):
  name: str
  variant: str

  def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
    ...


class MissingChunkIdError(RuntimeError):
  """Raised when a retrieved chunk has no chunk_id in its metadata.

  Deliberately fatal rather than scored as a miss. A collection ingested before
  deterministic ids would otherwise produce a complete, plausible-looking
  baseline that measures nothing -- the same shape of failure as the "0 internal
  citation edges" figure in CLAUDE.md, which meant "the lookup didn't run", not
  "there are no citations". One of those anecdotes is enough.
  """


class DenseChromaRetriever:
  """Phase 1 baseline: naive top-k dense similarity, exactly what the base repo does.

  Uses `similarity_search_with_score` rather than `as_retriever()` because the
  latter discards distances, which makes ties and near-misses uninspectable.
  """

  name = "dense_chroma"

  def __init__(self, provider: str = "groq", variant: str = "dense_baseline",
               stable_ties: bool = True):
    self.provider = provider
    self.variant = variant
    self.stable_ties = stable_ties
    self._vs = load_vectorstore(provider)     # built ONCE; get_embeddings reloads
                                              # MiniLM from disk on every call, so
                                              # constructing per query would mean
                                              # one model load per question.

  @property
  def collection(self):
    return self._vs._collection

  def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
    scored = self._vs.similarity_search_with_score(query, k=k)

    # Stable tie-break. With chunk_overlap=50 adjacent chunks are near-duplicates,
    # so near-ties in the top-k are common, and HNSW ordering among equal
    # distances is not guaranteed stable across runs. Unhandled, MRR jitters
    # between 1/r and 1/(r+1) and manufactures fake +/-0.01 "improvements"
    # between phases.
    if self.stable_ties:
      scored = sorted(scored, key=lambda ds: (ds[1], ds[0].metadata.get("chunk_id") or ""))

    out: List[RetrievedChunk] = []
    for rank, (doc, distance) in enumerate(scored, start=1):
      chunk_id = doc.metadata.get("chunk_id")
      if not chunk_id:
        raise MissingChunkIdError(
          "Retrieved a chunk with no metadata['chunk_id']. This collection predates "
          "deterministic chunk ids -- re-run `python scripts/ingest_corpus.py "
          f"--provider {self.provider} --reset`. Metadata was: {dict(doc.metadata)!r}"
        )
      out.append(RetrievedChunk(
        chunk_id=chunk_id,
        rank=rank,
        l2_distance=float(distance),
        page_content=doc.page_content,
        metadata=dict(doc.metadata),
      ))
    return out


class HybridRRFRetriever:
  """Phase 2: dense + BM25 fused with RRF, wrapped in the Phase 1 interface.

  All of the retrieval logic lives in server/core/hybrid_retriever.py -- the app
  and the eval must exercise the same code path, or the measured number is not a
  number about the shipped system. This class is only the adapter, and adding it
  changes ZERO lines of metrics.py, which is what the narrow
  `RetrieverUnderTest` protocol was built for in Phase 1.
  """

  name = "hybrid_rrf"

  def __init__(self, provider: str = "groq", variant: str = "hybrid_rrf60",
               rrf_k: int = None, pool: int = None,
               weights: Optional[Mapping[str, float]] = None):
    from core.hybrid_retriever import CANDIDATE_POOL, RRF_K, HybridRetriever

    self.provider = provider
    self.variant = variant
    self._impl = HybridRetriever(
      provider,
      rrf_k=RRF_K if rrf_k is None else rrf_k,
      pool=CANDIDATE_POOL if pool is None else pool,
      weights=weights,
    )

  @property
  def collection(self):
    return self._impl._vs._collection

  @property
  def stats(self) -> Mapping[str, Any]:
    return self._impl.stats

  def explain(self, query: str, chunk_id: str):
    return self._impl.explain(query, chunk_id)

  def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    for hit in self._impl.retrieve(query, k):
      chunk_id = hit.metadata.get("chunk_id")
      if not chunk_id:
        raise MissingChunkIdError(
          "Retrieved a chunk with no metadata['chunk_id']. This collection predates "
          "deterministic chunk ids -- re-run `python scripts/ingest_corpus.py "
          f"--provider {self.provider} --reset`. Metadata was: {dict(hit.metadata)!r}"
        )
      out.append(RetrievedChunk(
        chunk_id=chunk_id,
        rank=hit.rank,
        page_content=hit.page_content,
        metadata=dict(hit.metadata),
        l2_distance=hit.l2_distance,
        score=hit.rrf_score,
        ranks=dict(hit.ranks),
      ))
    return out


def to_ranking(chunks: Sequence[RetrievedChunk]) -> List[str]:
  """Ranked chunk ids, first occurrence wins."""
  seen: Set[str] = set()
  out: List[str] = []
  for c in chunks:
    if c.chunk_id and c.chunk_id not in seen:
      seen.add(c.chunk_id)
      out.append(c.chunk_id)
  return out


def found_by(chunk: RetrievedChunk) -> Optional[str]:
  """Which arm(s) surfaced this hit: "both" | "dense" | "bm25" | None.

  None for a single-arm retriever, which has no attribution to give. Aggregated
  over the gold chunks this is the most informative single table Phase 2
  produces: it separates "hybrid helped" from "hybrid helped BECAUSE BM25 found
  what dense could not", and it is what tells Phase 3 whether a reranker will
  have anything new to rerank.
  """
  if not chunk.ranks:
    return None
  dense = chunk.ranks.get("dense") is not None
  sparse = chunk.ranks.get("bm25") is not None
  if dense and sparse:
    return "both"
  if dense:
    return "dense"
  if sparse:
    return "bm25"
  return None


def count_near_ties(chunks: Sequence[RetrievedChunk], k: int = 10,
                    epsilon: float = 1e-6) -> int:
  """Adjacent pairs within the top-k whose distances are within epsilon.

  Recorded per run: a rising tie count means ranking is increasingly decided by
  the tie-break rather than by the retriever, which is worth knowing before
  attributing a delta to a phase.
  """
  top = list(chunks)[:k]
  return sum(1 for a, b in zip(top, top[1:])
             if a.l2_distance is not None and b.l2_distance is not None
             and abs(a.l2_distance - b.l2_distance) < epsilon)


def load_collection_index(provider: str = "groq") -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
  """One sweep of the collection -> (chunk_ids, id->text_sha8, id->page_content).

  Used to validate gold ids exist, to detect text drift, and to build the
  authoring tools. 2131 rows, so this is cheap and done once per run.
  """
  vs = load_vectorstore(provider)
  got = vs._collection.get(include=["metadatas", "documents"])

  ids = set(got["ids"])
  sha_by_id = {}
  text_by_id = {}
  for cid, meta, doc in zip(got["ids"], got["metadatas"], got["documents"]):
    sha_by_id[cid] = meta.get("text_sha8", "")
    text_by_id[cid] = doc

  malformed = [i for i in ids if not CHUNK_ID_RE.match(i)]
  if malformed:
    raise MissingChunkIdError(
      f"Collection contains {len(malformed)} ids that are not deterministic chunk ids "
      f"(e.g. {malformed[:3]}). Re-run scripts/ingest_corpus.py --reset."
    )
  return ids, sha_by_id, text_by_id


def read_ingest_manifest(provider: str = "groq") -> Optional[dict]:
  """The sidecar written at ingest; None if the collection predates it."""
  import json
  from pathlib import Path

  from config.settings import VECTORSTORE_DIRECTORY

  path = Path(VECTORSTORE_DIRECTORY[provider]) / "ingest_manifest.json"
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def assert_deterministic_retrieval(retriever: RetrieverUnderTest, query: str, k: int = 10) -> bool:
  """Same query twice must give the same ranking.

  HNSW search is approximate; if this ever fails, every cross-phase delta is
  suspect and should not be attributed to the phase.
  """
  first = to_ranking(retriever.retrieve(query, k))
  second = to_ranking(retriever.retrieve(query, k))
  if first != second:
    print(f"WARNING: retrieval is not deterministic for {query!r}", file=sys.stderr)
    print(f"  run 1: {first}", file=sys.stderr)
    print(f"  run 2: {second}", file=sys.stderr)
    return False
  return True
