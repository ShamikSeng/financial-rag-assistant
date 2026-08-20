"""Golden question set: schema, loading, and validation.

Pure stdlib, like metrics.py -- validation is testable without a collection
(pass `known_chunk_ids=None` to skip the collection-dependent rules).

Design note on `semantics`
--------------------------
The `semantics` field ("any_of" / "all_required") is an authored
**stratification label**. It is validated for consistency with the group
structure and used to split the report, but metrics.py NEVER branches on it --
scoring sees only the groups. That separation is what stops the two paths from
drifting apart: there is exactly one scoring code path, exercised by every
question regardless of label.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, FrozenSet, List, Mapping, Optional, Sequence, Tuple

CHUNK_ID_RE = re.compile(r"^(?P<arxiv_id>[^:]+)::p(?P<page>\d+)::c(?P<ordinal>\d+)$")

# Above this content-word Jaccard between a question and its gold chunk, the
# question is probably echoing the chunk's vocabulary rather than asking about
# it -- which flatters dense retrieval and shrinks the measurable lift from
# hybrid search and reranking.
LEAKAGE_JACCARD_WARN = 0.35

_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with without
is are was were be been being do does did doing have has had having it its as not no nor so such
what which who whom when where why how can could should would may might will shall about into
over under between during above below can't cannot each other more most some any all both
""".split())


class Severity(Enum):
  ERROR = "error"
  WARN = "warn"


@dataclass(frozen=True)
class ValidationIssue:
  severity: Severity
  code: str
  qid: Optional[str]
  message: str

  def __str__(self) -> str:
    where = f"[{self.qid}] " if self.qid else ""
    return f"{self.severity.value.upper():5s} {self.code:22s} {where}{self.message}"


@dataclass(frozen=True)
class GoldGroup:
  chunk_ids: Tuple[str, ...]
  note: Optional[str] = None


@dataclass(frozen=True)
class GoldenQuestion:
  qid: str
  question: str
  gold_groups: Tuple[GoldGroup, ...]
  semantics: str                        # "any_of" | "all_required"
  expected_answer: Optional[str] = None
  source_papers: Tuple[str, ...] = ()
  provenance: str = "llm_generated"     # "llm_generated" | "handwritten"
  tags: Tuple[str, ...] = ()
  chunk_text_sha8: Mapping[str, str] = field(default_factory=dict)

  @property
  def n_groups(self) -> int:
    return len(self.gold_groups)

  @property
  def group_sets(self) -> Tuple[FrozenSet[str], ...]:
    """The ONLY thing metrics.py ever sees."""
    return tuple(frozenset(g.chunk_ids) for g in self.gold_groups)

  @property
  def all_chunk_ids(self) -> Tuple[str, ...]:
    return tuple(cid for g in self.gold_groups for cid in g.chunk_ids)


@dataclass(frozen=True)
class GoldenSet:
  version: str
  corpus_fingerprint: str
  chunking: Mapping[str, Any]
  questions: Tuple[GoldenQuestion, ...]

  def __len__(self) -> int:
    return len(self.questions)


# --------------------------------------------------------------------------
# load / save
# --------------------------------------------------------------------------

def question_from_dict(raw: Mapping[str, Any]) -> GoldenQuestion:
  return GoldenQuestion(
    qid=raw["qid"],
    question=raw["question"],
    gold_groups=tuple(
      GoldGroup(chunk_ids=tuple(g["chunk_ids"]), note=g.get("note"))
      for g in raw["gold_groups"]
    ),
    semantics=raw["semantics"],
    expected_answer=raw.get("expected_answer"),
    source_papers=tuple(raw.get("source_papers", ())),
    provenance=raw.get("provenance", "llm_generated"),
    tags=tuple(raw.get("tags", ())),
    chunk_text_sha8=dict(raw.get("chunk_text_sha8", {})),
  )


def question_to_dict(q: GoldenQuestion) -> dict:
  return {
    "qid": q.qid,
    "question": q.question,
    "semantics": q.semantics,
    "gold_groups": [
      {"chunk_ids": list(g.chunk_ids), **({"note": g.note} if g.note else {})}
      for g in q.gold_groups
    ],
    "expected_answer": q.expected_answer,
    "source_papers": list(q.source_papers),
    "provenance": q.provenance,
    "tags": list(q.tags),
    "chunk_text_sha8": dict(q.chunk_text_sha8),
  }


def load_golden_set(path: Path) -> GoldenSet:
  raw = json.loads(Path(path).read_text(encoding="utf-8"))
  return GoldenSet(
    version=raw["version"],
    corpus_fingerprint=raw["corpus_fingerprint"],
    chunking=raw.get("chunking", {}),
    questions=tuple(question_from_dict(q) for q in raw["questions"]),
  )


def save_golden_set(gs: GoldenSet, path: Path) -> None:
  payload = {
    "version": gs.version,
    "corpus_fingerprint": gs.corpus_fingerprint,
    "chunking": dict(gs.chunking),
    "questions": [question_to_dict(q) for q in gs.questions],
  }
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# leakage measurement
# --------------------------------------------------------------------------

def content_words(text: str) -> FrozenSet[str]:
  words = re.findall(r"[a-z0-9]+", text.lower())
  return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def question_chunk_jaccard(question: str, chunk_text: str) -> float:
  """Content-word Jaccard between a question and its gold chunk.

  Measures generation bias rather than assuming it away. A question written by
  an LLM that was shown the chunk tends to reuse its distinctive vocabulary, so
  dense retrieval finds it trivially -- inflating the Phase 1 baseline and
  making Phases 2-3 look like they add nothing.
  """
  q_words = content_words(question)
  c_words = content_words(chunk_text)
  if not q_words or not c_words:
    return 0.0
  return len(q_words & c_words) / len(q_words | c_words)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_golden_set(
  gs: GoldenSet,
  known_chunk_ids: Optional[set] = None,
  known_text_sha8: Optional[Mapping[str, str]] = None,
  collection_fingerprint: Optional[str] = None,
  reporting_ks: Sequence[int] = (5, 10),
) -> List[ValidationIssue]:
  """Every rule carries a stable `code` so callers can filter deliberately."""
  issues: List[ValidationIssue] = []

  def err(code, qid, msg):
    issues.append(ValidationIssue(Severity.ERROR, code, qid, msg))

  def warn(code, qid, msg):
    issues.append(ValidationIssue(Severity.WARN, code, qid, msg))

  if collection_fingerprint is not None and gs.corpus_fingerprint != collection_fingerprint:
    err("FINGERPRINT_MISMATCH", None,
        f"golden set was authored against {gs.corpus_fingerprint[:23]}... but the "
        f"collection is {collection_fingerprint[:23]}... -- the corpus or the "
        f"chunking changed; gold ids may point at different text")

  seen_qids = set()
  min_k = min(reporting_ks) if reporting_ks else 5

  for q in gs.questions:
    if not q.qid:
      err("DUP_QID", None, "question has an empty qid")
      continue
    if q.qid in seen_qids:
      err("DUP_QID", q.qid, "duplicate qid")
    seen_qids.add(q.qid)

    if not q.gold_groups:
      err("EMPTY_GROUPS", q.qid, "question has no gold groups")
      continue

    # --- structure -------------------------------------------------------
    membership = {}
    for gi, group in enumerate(q.gold_groups):
      if not group.chunk_ids:
        err("EMPTY_GROUP", q.qid, f"gold group {gi} is empty")
        continue
      if len(set(group.chunk_ids)) != len(group.chunk_ids):
        warn("DUP_ID_IN_GROUP", q.qid, f"gold group {gi} repeats a chunk id")
      for cid in group.chunk_ids:
        if not CHUNK_ID_RE.match(cid):
          err("BAD_ID_FORMAT", q.qid, f"{cid!r} is not a valid chunk id")
        # Only CROSS-group repetition is an error -- that is what makes coverage
        # ranks non-distinct and breaks the DCG <= IDCG bound. A repeat inside a
        # single group is harmless (the group is a set) and is warned above.
        if cid in membership and membership[cid] != gi:
          err("GROUPS_OVERLAP", q.qid,
              f"{cid!r} appears in groups {membership[cid]} and {gi}; groups must be "
              f"pairwise disjoint (overlap breaks the DCG <= IDCG bound)")
        membership[cid] = gi

    # --- semantics label consistency -------------------------------------
    if q.semantics == "any_of":
      if q.n_groups != 1:
        err("SEMANTICS_MISMATCH", q.qid,
            f"semantics='any_of' implies exactly 1 gold group, found {q.n_groups}")
    elif q.semantics == "all_required":
      if q.n_groups < 2:
        err("SEMANTICS_MISMATCH", q.qid,
            f"semantics='all_required' implies >= 2 gold groups, found {q.n_groups} "
            f"(a single-hop question should be labelled 'any_of')")
    else:
      err("SEMANTICS_MISMATCH", q.qid, f"unknown semantics {q.semantics!r}")

    # --- the rule that matters most given chunk_overlap=50 ---------------
    if q.semantics == "all_required":
      located = []
      for group in q.gold_groups:
        for cid in group.chunk_ids:
          m = CHUNK_ID_RE.match(cid)
          if m:
            located.append((m.group("arxiv_id"), int(m.group("page")), int(m.group("ordinal"))))
            break
      for i in range(len(located)):
        for j in range(i + 1, len(located)):
          a, b = located[i], located[j]
          if a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) <= 1:
            warn("SUSPICIOUS_MULTIHOP", q.qid,
                 f"groups {i} and {j} are adjacent chunks on the same page "
                 f"({a[0]} p{a[1]} c{a[2]}/c{b[2]}); with chunk_overlap=50 these share "
                 f"text and are probably 'any_of' alternatives, not distinct hops")

    if q.n_groups > min_k:
      warn("N_GROUPS_EXCEEDS_K", q.qid,
           f"{q.n_groups} gold groups but smallest reporting k is {min_k}; "
           f"Recall@{min_k} is capped at {min_k}/{q.n_groups}")

    # --- collection-dependent rules --------------------------------------
    if known_chunk_ids is not None:
      for cid in q.all_chunk_ids:
        if cid not in known_chunk_ids:
          err("UNKNOWN_CHUNK_ID", q.qid,
              f"{cid!r} is not in the collection (typo, or stale after re-chunking)")

    if known_text_sha8 is not None:
      for cid, expected in q.chunk_text_sha8.items():
        actual = known_text_sha8.get(cid)
        if actual is not None and actual != expected:
          err("TEXT_DRIFT", q.qid,
              f"{cid!r} text hash changed ({expected} -> {actual}); the id still "
              f"resolves but now points at different text")

  return issues


def has_errors(issues: Sequence[ValidationIssue]) -> bool:
  return any(i.severity is Severity.ERROR for i in issues)
