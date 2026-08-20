"""Put server/ on sys.path for the chunk_ids tests.

Deliberately NO chdir here. server/core/chunk_ids.py imports only stdlib and
needs no cwd, so these tests stay as fast and side-effect-free as the metrics
tests. Only modules that touch the vectorstore need eval._bootstrap.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "server"

for path in (str(REPO_ROOT), str(SERVER_DIR)):
  if path not in sys.path:
    sys.path.insert(0, path)
