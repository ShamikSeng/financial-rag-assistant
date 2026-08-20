"""Chunk-ID derivation: determinism, ordinal semantics, path portability.

Determinism is the whole basis of the golden set -- if assign_chunk_ids is not a
pure function of the ordered chunk list, every gold id is provisional.
"""

from types import SimpleNamespace

import pytest

from core.chunk_ids import (
  CHUNK_ID_RE,
  assign_chunk_ids,
  basename_stem,
  make_chunk_id,
  parse_chunk_id,
  text_sha8,
)


def mk(source, page, text="body text"):
  return SimpleNamespace(metadata={"source": source, "page": page}, page_content=text)


# ------------------------------------------------------------ path parsing --

def test_basename_stem_handles_windows_paths():
  """The collection was ingested on Windows, so `source` holds backslash paths.
  PurePosixPath would return the whole mangled string unparsed."""
  assert basename_stem(r"C:\Users\a_sen\Downloads\repo\data\papers\raw\2004.04906.pdf") == "2004.04906"


def test_basename_stem_handles_posix_paths():
  assert basename_stem("/home/x/data/papers/raw/2404.16130.pdf") == "2404.16130"


def test_basename_stem_case_insensitive_extension():
  assert basename_stem("/x/2004.04906.PDF") == "2004.04906"


def test_basename_stem_leaves_arxiv_version_suffix_alone():
  # freeze_corpus.py writes unversioned filenames, but be explicit about it
  assert basename_stem("/x/2004.04906v2.pdf") == "2004.04906v2"


# -------------------------------------------------------------- round-trip --

def test_make_parse_round_trip():
  cid = make_chunk_id("2004.04906", 12, 3)
  assert cid == "2004.04906::p12::c3"
  assert parse_chunk_id(cid) == ("2004.04906", 12, 3)
  assert CHUNK_ID_RE.match(cid)


def test_parse_rejects_malformed():
  for bad in ["2004.04906", "2004.04906::p1", "2004.04906::pX::c1", "a::p1::c1::extra", ""]:
    with pytest.raises(ValueError):
      parse_chunk_id(bad)


def test_make_rejects_colon_in_arxiv_id():
  with pytest.raises(ValueError):
    make_chunk_id("bad::id", 0, 0)


def test_make_rejects_negative():
  with pytest.raises(ValueError):
    make_chunk_id("2004.04906", -1, 0)


# --------------------------------------------------------- T20: assignment --

def test_T20_ordinal_resets_per_page_and_per_paper():
  chunks = [
    mk("/x/X.pdf", 0), mk("/x/X.pdf", 0), mk("/x/X.pdf", 0),
    mk("/x/X.pdf", 1),
    mk("/x/Y.pdf", 0),
  ]
  assert assign_chunk_ids(chunks) == [
    "X::p0::c0", "X::p0::c1", "X::p0::c2",
    "X::p1::c0",
    "Y::p0::c0",
  ]


def test_T20_ids_are_unique():
  chunks = [mk("/x/X.pdf", p) for p in range(20) for _ in range(3)]
  ids = assign_chunk_ids(chunks)
  assert len(set(ids)) == len(ids) == 60


def test_T20_determinism_across_runs():
  def build():
    return [mk("/x/X.pdf", 0), mk("/x/X.pdf", 0), mk("/x/Y.pdf", 3)]
  assert assign_chunk_ids(build()) == assign_chunk_ids(build())


def test_assign_writes_metadata_mirror():
  """metadata['chunk_id'] is the copy that actually survives retrieval --
  langchain_community rebuilds Documents from page_content + metadata and never
  reads Chroma's ids column."""
  chunks = [mk("/x/2004.04906.pdf", 2, "hello")]
  ids = assign_chunk_ids(chunks)
  meta = chunks[0].metadata
  assert meta["chunk_id"] == ids[0] == "2004.04906::p2::c0"
  assert meta["arxiv_id"] == "2004.04906"
  assert meta["text_sha8"] == text_sha8("hello")


def test_assign_rejects_missing_source():
  with pytest.raises(ValueError, match="source"):
    assign_chunk_ids([SimpleNamespace(metadata={"page": 0}, page_content="x")])


def test_assign_rejects_missing_page():
  with pytest.raises(ValueError, match="page"):
    assign_chunk_ids([SimpleNamespace(metadata={"source": "/x/A.pdf"}, page_content="x")])


def test_page_is_zero_indexed():
  """pypdf's `page` is 0-indexed, so p0 is the FIRST page. Pinned because
  anyone reading a PDF viewer will assume otherwise."""
  chunks = [mk("/x/X.pdf", 0)]
  assert assign_chunk_ids(chunks) == ["X::p0::c0"]


def test_text_sha8_is_stable_and_short():
  assert text_sha8("abc") == text_sha8("abc")
  assert len(text_sha8("abc")) == 8
  assert text_sha8("abc") != text_sha8("abd")
