import os

from typing import List, Optional
from fastapi import UploadFile

from config.settings import (
  DEFAULT_RETRIEVAL_MODE,
  GOOGLE_API_KEY,
  MODEL_OPTIONS,
  RETRIEVAL_MODES,
  VECTORSTORE_DIRECTORY,
)
from core.document_processor import save_uploaded_file, load_documents_from_paths, split_documents_to_chunks

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.logger import logger

# LangChain's own similarity_search default, made explicit so it is visible.
# NOTE: build_llm_chain() in llm_chain_factory.py retrieves k=3, so the search
# endpoint and the chat path do not retrieve the same set. That inconsistency is
# deliberately left alone here -- unifying it is a *ranking* change, and making
# it before the Phase 1 baseline is recorded would bake the fix into the
# reference point and lose any clean before/after for it. Tracked in
# PROJECT_LOG.md's Open questions/TODOs.
DEFAULT_SEARCH_K = 4


def vectorstore_exists(persist_path: str) -> bool:
  exists = os.path.exists(persist_path) and bool(os.listdir(persist_path))
  logger.debug(f"Vectorstore exists at {persist_path}: {exists}")
  return exists

def get_embeddings(model_provider: str):
  logger.debug(f"Getting embeddings for provider: {model_provider}")
  if model_provider == "groq":
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
  elif model_provider == "gemini":
    return GoogleGenerativeAIEmbeddings(
      model="models/gemini-embedding-001",
      google_api_key=GOOGLE_API_KEY
    )
  else:
    logger.error(f"Unsupported LLM Provider: {model_provider}")
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

def initialize_empty_vectorstores():
  logger.info("Initializing empty vectorstores...")
  for provider in MODEL_OPTIONS.keys():
    persist_path = VECTORSTORE_DIRECTORY[provider]
    os.makedirs(persist_path, exist_ok=True)

    if not os.listdir(persist_path):
      embedding = get_embeddings(provider)
      Chroma(
        embedding_function=embedding,
        persist_directory=persist_path
      )
      logger.debug(f"Initialized vectorstore for {provider} at {persist_path}")

  logger.info("Vectorstore initialization complete.")

async def upsert_vectorstore_from_pdfs(uploaded_files: List[UploadFile], model_provider: str):
  logger.debug(f"Upserting vectorstore for {model_provider}")
  file_paths = await save_uploaded_file(uploaded_files)
  docs = load_documents_from_paths(file_paths)
  chunks = split_documents_to_chunks(docs)
  embedding = get_embeddings(model_provider)

  persist_path = VECTORSTORE_DIRECTORY[model_provider]

  if vectorstore_exists(persist_path):
    logger.debug("Appending to existing vectorstore...")
    vectorstore = Chroma(persist_directory=persist_path, embedding_function=embedding)
    vectorstore.add_documents(chunks)
    logger.debug(f"Added {len(chunks)} chunks to existing vectorstore.")
  else:
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_path)
    logger.debug(f"Created new vectorstore with {len(chunks)} chunks.")

  # The BM25 arm is an in-memory snapshot of this collection, so it is now
  # stale. Left uninvalidated, the hybrid path would fuse a dense arm that sees
  # the upload with a sparse arm that does not -- two different corpora, one
  # confident-looking answer.
  from core.hybrid_retriever import reset_hybrid_cache
  reset_hybrid_cache(model_provider)

  return vectorstore

def load_vectorstore(model_provider: str):
  persist_path = VECTORSTORE_DIRECTORY[model_provider]
  logger.debug(f"Loading vectorstore from {persist_path}")

  if vectorstore_exists(persist_path):
    logger.debug(f"Loading existing vectorstore for provider: {model_provider}")
    return Chroma(persist_directory=persist_path, embedding_function=get_embeddings(model_provider))

  logger.debug(f"VectorStore not found for provider: {model_provider}")
  raise ValueError(f"VectorStore not found for provider: {model_provider}")

def get_collections_count(model_provider: str):
  logger.debug(f"Getting collection count for provider: {model_provider}")
  vectorstore = load_vectorstore(model_provider)
  return vectorstore._collection.count()

def resolve_retrieval_mode(mode: Optional[str] = None) -> str:
  """None -> the configured default. Anything unrecognised is a hard error.

  Silently falling back to dense on a typo would report hybrid-shaped results
  from the dense path, which is the exact class of failure this project keeps
  legislating against.
  """
  resolved = (mode or DEFAULT_RETRIEVAL_MODE).lower()
  if resolved not in RETRIEVAL_MODES:
    raise ValueError(f"Unsupported retrieval mode: {resolved!r} (expected one of {RETRIEVAL_MODES})")
  return resolved


def find_similar_chunks(model_provider: str, query: str, k: int = DEFAULT_SEARCH_K,
                        mode: Optional[str] = None):
  mode = resolve_retrieval_mode(mode)
  logger.debug(f"Searching for similar chunks for provider: {model_provider} (k={k}, mode={mode})")

  if mode == "hybrid":
    from core.hybrid_retriever import get_hybrid_retriever
    return get_hybrid_retriever(model_provider).as_documents(query, k)

  vectorstore = load_vectorstore(model_provider)
  return vectorstore.similarity_search(query, k=k)

def find_similar_chunks_with_scores(model_provider: str, query: str, k: int = DEFAULT_SEARCH_K):
  """Same retrieval, but keeping the distances the eval harness ranks on.

  Returns List[Tuple[Document, float]] where the float is Chroma's **squared L2
  distance -- lower is better**. It is not a similarity and must not be treated
  as one: the collection is built with the default `hnsw:space=l2`, and
  HuggingFaceEmbeddings is constructed here without `encode_kwargs`, so vectors
  are unnormalized and the usual `L2^2 = 2 - 2*cos` identity does not hold.

  Deliberately not `similarity_search_with_relevance_scores`, whose Euclidean
  relevance function is `1 - d/sqrt(2)` and assumes unit vectors -- against this
  collection it emits large negative "similarities".

  Dense-only by design: an RRF score is not a distance and the two must not be
  returned under one field name. The hybrid equivalent is
  find_hybrid_chunks_with_details(), which returns fusion scores and per-arm
  ranks under their own names.
  """
  logger.debug(f"Scored search for provider: {model_provider} (k={k})")
  vectorstore = load_vectorstore(model_provider)
  return vectorstore.similarity_search_with_score(query, k=k)


def find_hybrid_chunks_with_details(model_provider: str, query: str, k: int = DEFAULT_SEARCH_K):
  """Hybrid retrieval with the fusion detail kept: List[FusedHit].

  Each hit carries its RRF score plus the rank it held in each arm (None where
  an arm never surfaced it), which is what makes "BM25 found this and dense did
  not" inspectable from the API rather than only in the eval artifacts.
  """
  from core.hybrid_retriever import get_hybrid_retriever
  logger.debug(f"Hybrid search for provider: {model_provider} (k={k})")
  return get_hybrid_retriever(model_provider).retrieve(query, k)
