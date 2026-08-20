import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TEMPFILE_UPLOAD_DIRECTORY = "./temp/uploaded_files"

MODEL_OPTIONS = {
  "groq": {
    "playground": "https://console.groq.com",
    "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
  },
  "gemini": {
    "playground": "https://ai.google.dev",
    "models": ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
  }
}

VECTORSTORE_DIRECTORY = {
  key.lower(): f"./data/{key.lower()}_vector_store"
  for key in MODEL_OPTIONS.keys()
}

# Which retrieval path the search endpoint and the chat chain use by default.
# "dense"  -- Phase 0/1 behaviour: naive top-k Chroma similarity.
# "hybrid" -- Phase 2: dense + BM25 fused with RRF (core/hybrid_retriever.py).
#
# The default is NOT a taste call. It ships as whichever variant the eval
# harness showed to be better under the decision rule pre-registered in
# PROJECT_LOG.md before the numbers were seen (paired bootstrap on nDCG@10, CI
# lower bound above zero, no stratum regressing). Callers can override per
# request via `retrieval_mode`, and RETRIEVAL_MODE in the environment overrides
# the default without a code change.
# Flipped to "hybrid" on 2026-08-20 because the rule fired, not because hybrid
# was the newer code: pooled nDCG@10 +0.1532, 95% paired-bootstrap CI
# [+0.0971, +0.2099], n=89, no stratum regressing. See the Eval Results table.
RETRIEVAL_MODES = ("dense", "hybrid")
DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
