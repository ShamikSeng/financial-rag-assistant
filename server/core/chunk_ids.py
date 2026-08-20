"""Deterministic chunk identity for the frozen corpus.

Why this exists
---------------
Chunks originally entered Chroma with random UUID4 ids, and langchain_community
0.4.2 drops even those on the way back out: `_results_to_docs_and_scores` builds
`Document(page_content=..., metadata=...)` and never reads the `ids` column. So a
retrieved chunk had no identity at all, which makes an eval golden set
("expected_chunk_ids") impossible to express, and makes Phase 2's RRF fusion and
Phase 3's reranking impossible to implement -- both of those join *ranked lists
of documents*, which requires documents to have names.

The scheme: `{arxiv_id}::p{page}::c{ordinal}`, e.g. `2004.04906::p2::c1`.

Two properties matter:

1. **It is written in BOTH places.** The id is passed to Chroma as `ids=` (which
   makes re-ingest idempotent, since Chroma upserts by id) *and* stored in
   `metadata["chunk_id"]` (which is the only one of the two that survives
   retrieval, per the langchain_community behaviour above). Neither alone is
   sufficient.

2. **It is positional, not content-derived.** A content hash would be equally
   stable but tells you nothing when it breaks. Phase 5 (multimodal ingestion)
   will re-chunk the corpus, and when that happens a positional id still says
   "page 2 of DPR", so a stale gold id can be re-resolved at page granularity by
   hand. A stale content hash is unrecoverable.

Determinism requires all three of: the sorted PDF path list, the same pypdf page
segmentation, and the same TokenTextSplitter(chunk_size=500, chunk_overlap=50).
`CHUNKING_VERSION` and `corpus_fingerprint()` exist to make a violation of that
loud rather than silent -- see the "0 internal citation edges" lesson in
CLAUDE.md, where a pipeline that hadn't run looked exactly like a pipeline that
had run and found nothing.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence

CHUNK_ID_RE = re.compile(r"^(?P<arxiv_id>[^:]+)::p(?P<page>\d+)::c(?P<ordinal>\d+)$")

# Bump on ANY change to splitter params, splitter class, or PDF loader. This
# value feeds corpus_fingerprint(), so bumping it invalidates every golden set
# built against the old chunking -- which is the point.
CHUNKING_VERSION = "v1"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class ChunkIdParts(NamedTuple):
  arxiv_id: str
  page: int
  ordinal: int


def basename_stem(source: str) -> str:
  """'C:\\Users\\...\\raw\\2004.04906.pdf' -> '2004.04906'.

  Splits on BOTH separators deliberately. The `source` metadata written by
  PyPDFLoader is an absolute path in the OS-native form of whichever machine ran
  the ingest, so a Windows-produced collection holds backslash paths that
  PurePosixPath would return unparsed (`.stem` on the whole mangled string).
  Writing `arxiv_id` into metadata at ingest is what ultimately frees the
  harness from having to parse `source` at all.
  """
  normalized = source.replace("\\", "/")
  name = normalized.rsplit("/", 1)[-1]
  return name[:-4] if name.lower().endswith(".pdf") else name


def make_chunk_id(arxiv_id: str, page: int, ordinal: int) -> str:
  if "::" in arxiv_id or ":" in arxiv_id:
    raise ValueError(f"arxiv_id must not contain ':' -- got {arxiv_id!r}")
  if page < 0 or ordinal < 0:
    raise ValueError(f"page and ordinal must be non-negative -- got page={page}, ordinal={ordinal}")
  return f"{arxiv_id}::p{page}::c{ordinal}"


def parse_chunk_id(chunk_id: str) -> ChunkIdParts:
  match = CHUNK_ID_RE.match(chunk_id)
  if not match:
    raise ValueError(f"Malformed chunk_id: {chunk_id!r} (expected '{{arxiv_id}}::p{{page}}::c{{ordinal}}')")
  return ChunkIdParts(
    arxiv_id=match.group("arxiv_id"),
    page=int(match.group("page")),
    ordinal=int(match.group("ordinal")),
  )


def text_sha8(text: str) -> str:
  """Short content hash, stored per chunk as a drift tripwire.

  Not an identifier -- the chunk_id is the identifier. This is the thing that
  catches "the id still resolves but now points at different text", which is
  exactly what a pypdf or langchain-text-splitters upgrade produces and which
  otherwise runs to completion and reports plausible numbers.
  """
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def assign_chunk_ids(chunks: Sequence[Any]) -> List[str]:
  """Assign deterministic ids to an ORDERED chunk list, mutating metadata.

  `ordinal` is a running counter within (arxiv_id, page), reset per page. Note
  that `page` here is pypdf's 0-indexed value -- `p0` is the first page of the
  PDF, not the second. (`page_label`, which PyPDFLoader also sets, is the
  human-facing 1-indexed string; it is deliberately not used here because it is
  a free-form label and some papers have roman-numeral front matter.)

  Writes metadata['chunk_id'], ['arxiv_id'] and ['text_sha8'] onto each chunk.
  Returns the ids in the same order, for passing to Chroma as `ids=`.
  """
  counters: Dict[tuple, int] = {}
  ids: List[str] = []

  for chunk in chunks:
    source = chunk.metadata.get("source")
    if not source:
      raise ValueError(f"Chunk is missing 'source' metadata; cannot derive arxiv_id: {chunk.metadata!r}")

    arxiv_id = basename_stem(source)
    page = chunk.metadata.get("page")
    if page is None:
      raise ValueError(f"Chunk is missing 'page' metadata: {chunk.metadata!r}")
    page = int(page)

    key = (arxiv_id, page)
    ordinal = counters.get(key, 0)
    counters[key] = ordinal + 1

    chunk_id = make_chunk_id(arxiv_id, page, ordinal)
    chunk.metadata["chunk_id"] = chunk_id
    chunk.metadata["arxiv_id"] = arxiv_id
    chunk.metadata["text_sha8"] = text_sha8(chunk.page_content)
    ids.append(chunk_id)

  if len(set(ids)) != len(ids):
    raise ValueError("assign_chunk_ids produced duplicate ids -- chunk list was not in a stable order")

  return ids


def corpus_fingerprint(manifest_path: Path) -> str:
  """Fingerprint over (chunking config, frozen corpus contents).

  Recomputed at eval time and compared against the value recorded at ingest. A
  mismatch means the collection being measured is not the collection the golden
  set was authored against -- either the corpus changed (violating the FROZEN
  non-negotiable in PROJECT_LOG.md) or the chunking did.
  """
  manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
  papers = sorted(
    (p["arxiv_id"], p.get("sha256", "")) for p in manifest["papers"]
  )
  payload = json.dumps(
    {
      "chunking_version": CHUNKING_VERSION,
      "chunk_size": CHUNK_SIZE,
      "chunk_overlap": CHUNK_OVERLAP,
      "papers": papers,
    },
    sort_keys=True,
  )
  return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
