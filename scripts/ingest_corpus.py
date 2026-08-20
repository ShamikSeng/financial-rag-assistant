"""
One-off ingestion script for the frozen 45-paper corpus (data/papers/raw/, see
data/papers/corpus_manifest.json for provenance).

Reuses the existing document_processor.py -> vector_database.py pipeline
(PyPDFLoader -> TokenTextSplitter chunking -> provider embeddings -> Chroma
upsert) rather than duplicating any of that logic. The only thing skipped is
the FastAPI-upload-specific save_uploaded_file/validate_pdf step, since the
corpus PDFs already live on disk and are already provenance-tracked in the
manifest.

Chunks are assigned deterministic ids ({arxiv_id}::p{page}::c{ordinal}, see
core/chunk_ids.py) rather than Chroma's default random UUID4s. This makes
re-ingest an idempotent upsert, gives the Phase 1 eval golden set something
stable to point at, and gives Phase 2's RRF fusion / Phase 3's reranker the
document identity they need to join ranked lists.

Usage (from repo root, with `pip install -e .` done and .env populated):
    python scripts/ingest_corpus.py --provider groq
    python scripts/ingest_corpus.py --provider gemini
    python scripts/ingest_corpus.py --provider both
    python scripts/ingest_corpus.py --provider groq --reset   # wipe and rebuild
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(SERVER_DIR))

# VECTORSTORE_DIRECTORY in config/settings.py is a *relative* path ("./data/{provider}_vector_store"),
# resolved against the process cwd. The app is normally run via `cd server && uvicorn main:app`, so
# that relative path means server/data/{provider}_vector_store at runtime -- not repo-root/data/.
# chdir here so this script writes to the exact same place the running app will read from.
# (python-dotenv's load_dotenv() still finds the repo-root .env by walking up parent dirs.)
os.chdir(SERVER_DIR)

from core.chunk_ids import (  # noqa: E402
  CHUNK_OVERLAP,
  CHUNK_SIZE,
  CHUNKING_VERSION,
  assign_chunk_ids,
  corpus_fingerprint,
)
from core.document_processor import load_documents_from_paths, split_documents_to_chunks  # noqa: E402
from core.vector_database import get_embeddings, vectorstore_exists, VECTORSTORE_DIRECTORY  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "papers" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "papers" / "corpus_manifest.json"
BATCH_SIZE = 100
GEMINI_BATCH_SIZE = 20
GEMINI_BATCH_DELAY_SECONDS = 2

EMBEDDING_MODELS = {
  "groq": "sentence-transformers/all-MiniLM-L12-v2",
  "gemini": "models/gemini-embedding-001",
}


def library_versions() -> dict:
  """Versions of the packages that can silently change chunking or embedding.

  pyproject.toml pins nothing, so a transparent upgrade of pypdf or
  langchain-text-splitters would re-segment the corpus and invalidate every gold
  chunk id while every run still completed successfully. Recording them here at
  least makes that drift *detectable* after the fact. (Actually pinning them is
  tracked in PROJECT_LOG.md's Open questions/TODOs.)
  """
  from importlib.metadata import PackageNotFoundError, version

  packages = ["pypdf", "langchain-text-splitters", "langchain-community", "chromadb", "sentence-transformers"]
  out = {}
  for pkg in packages:
    try:
      out[pkg] = version(pkg)
    except PackageNotFoundError:
      out[pkg] = None
  return out


def ingest(provider: str, reset: bool = False) -> int:
  pdf_paths = sorted(str(p) for p in RAW_DIR.glob("*.pdf"))
  print(f"[{provider}] Found {len(pdf_paths)} PDFs in {RAW_DIR}")
  if not pdf_paths:
    raise SystemExit(f"No PDFs found in {RAW_DIR}")

  persist_path = VECTORSTORE_DIRECTORY[provider]

  if reset and Path(persist_path).exists():
    print(f"[{provider}] --reset: deleting existing collection at {persist_path}")
    shutil.rmtree(persist_path)

  docs = load_documents_from_paths(pdf_paths)
  print(f"[{provider}] Loaded {len(docs)} page-documents")

  chunks = split_documents_to_chunks(docs)
  print(f"[{provider}] Split into {len(chunks)} chunks "
        f"(TokenTextSplitter, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

  # Deterministic ids, written both as Chroma's primary key (making re-ingest an
  # idempotent upsert rather than a duplicate-append) and into metadata (the only
  # one of the two that survives retrieval -- langchain_community rebuilds
  # Documents from page_content + metadata and never reads the ids column).
  chunk_ids = assign_chunk_ids(chunks)
  print(f"[{provider}] Assigned {len(set(chunk_ids))} unique deterministic chunk ids "
        f"(e.g. {chunk_ids[0]})")

  embedding = get_embeddings(provider)
  Path(persist_path).mkdir(parents=True, exist_ok=True)

  already_exists = vectorstore_exists(persist_path)
  vectorstore = Chroma(persist_directory=persist_path, embedding_function=embedding)

  start_idx = vectorstore._collection.count() if already_exists else 0
  if start_idx:
    print(f"[{provider}] Existing vectorstore has {start_idx} chunks already; resuming from chunk {start_idx}.")

  batch_size = GEMINI_BATCH_SIZE if provider == "gemini" else BATCH_SIZE
  batch_delay = GEMINI_BATCH_DELAY_SECONDS if provider == "gemini" else 0

  total = len(chunks)
  for i in range(start_idx, total, batch_size):
    batch = chunks[i:i + batch_size]
    batch_ids = chunk_ids[i:i + batch_size]
    t0 = time.time()
    vectorstore.add_documents(batch, ids=batch_ids)
    dt = time.time() - t0
    print(f"[{provider}] Embedded+upserted batch {i}-{i + len(batch)}/{total} ({dt:.1f}s)")
    if batch_delay:
      time.sleep(batch_delay)

  count = vectorstore._collection.count()
  print(f"[{provider}] Done. Vectorstore now has {count} chunks at {persist_path}")

  write_ingest_manifest(provider, persist_path, count, len(docs), len(pdf_paths))
  return count


def write_ingest_manifest(provider: str, persist_path: str, chunk_count: int,
                          page_doc_count: int, pdf_count: int) -> None:
  """Sidecar recording what this collection actually contains.

  The eval harness compares its own recomputed corpus_fingerprint against this
  file and aborts on mismatch, so that "the golden set was authored against a
  different collection than the one being measured" is a hard error rather than
  a quietly wrong number.
  """
  manifest = {
    "provider": provider,
    "ingested_at": datetime.now(timezone.utc).isoformat(),
    "corpus_fingerprint": corpus_fingerprint(MANIFEST_PATH),
    "chunking_version": CHUNKING_VERSION,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "pdf_count": pdf_count,
    "page_doc_count": page_doc_count,
    "chunk_count": chunk_count,
    "embedding_model": EMBEDDING_MODELS[provider],
    "library_versions": library_versions(),
  }
  out_path = Path(persist_path) / "ingest_manifest.json"
  out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  print(f"[{provider}] Wrote {out_path} (fingerprint {manifest['corpus_fingerprint'][:23]}...)")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--provider", choices=["groq", "gemini", "both"], default="groq")
  parser.add_argument(
    "--reset",
    action="store_true",
    help="Delete the existing collection before ingesting. Required to re-ingest a "
         "populated collection; without it, a non-empty collection is treated as a "
         "resume and only the missing tail is added.",
  )
  args = parser.parse_args()

  providers = ["groq", "gemini"] if args.provider == "both" else [args.provider]
  results = {}
  for provider in providers:
    results[provider] = ingest(provider, reset=args.reset)

  print("\n=== Summary ===")
  for provider, count in results.items():
    print(f"{provider}: {count} chunks")


if __name__ == "__main__":
  main()
