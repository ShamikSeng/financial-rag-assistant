"""Import shim for reaching the server package from the eval harness.

`VECTORSTORE_DIRECTORY` in server/config/settings.py is a *relative* path
("./data/{provider}_vector_store") resolved against the process cwd. The app is
run as `cd server && uvicorn main:app`, so at runtime those paths mean
server/data/... . A harness running from the repo root would instead resolve
them to repo_root/data/ -- which exists and is non-empty (it holds the paper
PDFs), so `vectorstore_exists()` returns True and you get a Chroma handle
pointed at a directory with no collection. Silently wrong, not a clean error.

So: chdir into server/, exactly as scripts/ingest_corpus.py already does.

CONSEQUENCE to design around: after bootstrap, every relative path resolves
under server/. Callers must resolve their own paths against REPO_ROOT *before*
bootstrapping, or `--golden-set eval/data/golden_set.v1.json` silently becomes
server/eval/data/... .

eval/metrics.py deliberately does not import this module -- keeping the metrics
pure is what lets its tests run without any of the above.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"

_BOOTSTRAPPED = False


def bootstrap_server_context() -> Path:
  """Put server/ on sys.path and chdir into it. Idempotent. Returns prior cwd."""
  global _BOOTSTRAPPED
  previous = Path.cwd()

  if not _BOOTSTRAPPED:
    for path in (str(REPO_ROOT), str(SERVER_DIR)):
      if path not in sys.path:
        sys.path.insert(0, path)
    _BOOTSTRAPPED = True

  if Path.cwd() != SERVER_DIR:
    os.chdir(SERVER_DIR)

  assert Path.cwd() == SERVER_DIR, f"chdir failed: cwd is {Path.cwd()}, expected {SERVER_DIR}"
  return previous


@contextmanager
def server_context() -> Iterator[Path]:
  """Same, but restores cwd on exit. Use anywhere pytest might interleave."""
  previous = bootstrap_server_context()
  try:
    yield SERVER_DIR
  finally:
    os.chdir(previous)
