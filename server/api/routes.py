from fastapi import APIRouter, UploadFile, File, Form

from config.settings import MODEL_OPTIONS
from core.vector_database import (
    DEFAULT_SEARCH_K,
    get_collections_count,
    find_hybrid_chunks_with_details,
    find_similar_chunks,
    find_similar_chunks_with_scores,
    resolve_retrieval_mode,
    upsert_vectorstore_from_pdfs,
    load_vectorstore
)
from core.llm_chain_factory import build_llm_chain
from api.schemas import SearchQueryRequest, ChatRequest, StandardAPIResponse
from utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=StandardAPIResponse)
def health_check():
  logger.debug("Health check requested")
  return StandardAPIResponse(
    status="success",
    data="ok",
    message="Service is healthy"
  )

@router.get("/llm", response_model=StandardAPIResponse)
async def get_llm_options():
  logger.debug("Fetching LLM providers.")
  return StandardAPIResponse(
    status="success",
    data=[provider.title() for provider in MODEL_OPTIONS.keys()]
  )

@router.get("/llm/{model_provider}", response_model=StandardAPIResponse)
async def get_llm_models(model_provider: str):
  model_provider = model_provider.lower()
  if model_provider not in MODEL_OPTIONS:
    logger.warning(f"Invalid model provider: {model_provider}")
    return StandardAPIResponse(status="error", message="Invalid model provider.")

  logger.debug(f"Fetching models for provider: {model_provider}")
  return StandardAPIResponse(
    status="success",
    data=MODEL_OPTIONS[model_provider]["models"]
  )

@router.post("/upload_and_process_pdfs", response_model=StandardAPIResponse)
async def upload_and_process_pdfs(
  files: list[UploadFile] = File(...),
  model_provider: str = Form(...)
):
  try:
    model_provider = model_provider.lower()
    logger.info(f"Received {len(files)} files for model provider: {model_provider}")
    await upsert_vectorstore_from_pdfs(files, model_provider)
    logger.info("Files processed successfully")
    return StandardAPIResponse(status="success", data="PDFs processed successfully.")
  except Exception as e:
    logger.exception("Error while uploading and processing files")
    return StandardAPIResponse(status="error", message=str(e))

@router.get("/vector_store/count/{model_provider}", response_model=StandardAPIResponse)
async def get_vectorstore_count(model_provider: str):
  try:
    model_provider = model_provider.lower()
    logger.info(f"Getting collection count for provider: {model_provider}")
    count = get_collections_count(model_provider)
    return StandardAPIResponse(status="success", data=count)
  except Exception as e:
    logger.exception("Error getting collection count")
    return StandardAPIResponse(status="error", message=str(e))

@router.post("/vector_store/search", response_model=StandardAPIResponse)
async def get_vectorstore_search(request: SearchQueryRequest):
  try:
    model_provider = request.model_provider.lower()
    k = request.k if request.k is not None else DEFAULT_SEARCH_K
    mode = resolve_retrieval_mode(request.retrieval_mode)
    logger.info(f"Search requested with query: {request.query} for provider: "
                f"{request.model_provider} (k={k}, mode={mode})")

    if request.include_scores and mode == "hybrid":
      # Deliberately NOT `l2_distance`: an RRF score is a fused rank statistic,
      # higher is better, and reusing the distance field name for it would be a
      # unit error waiting to be read as one. The per-arm ranks are included
      # because "BM25 found this at 4, dense never did" is the single most
      # useful thing to see when inspecting a hybrid result.
      results = [
        {
          "document": {"page_content": hit.page_content, "metadata": dict(hit.metadata)},
          "rrf_score": hit.rrf_score,
          "dense_rank": hit.ranks.get("dense"),
          "sparse_rank": hit.ranks.get("bm25"),
          "l2_distance": hit.l2_distance,
          "bm25_score": hit.bm25_score,
          "found_by": hit.found_by,
        }
        for hit in find_hybrid_chunks_with_details(model_provider, request.query, k=k)
      ]
    elif request.include_scores:
      scored = find_similar_chunks_with_scores(model_provider, request.query, k=k)
      # `l2_distance`, not `score` -- squared L2, lower is better. See the
      # docstring on find_similar_chunks_with_scores.
      results = [{"document": doc, "l2_distance": dist} for doc, dist in scored]
    else:
      results = find_similar_chunks(model_provider, request.query, k=k, mode=mode)

    return StandardAPIResponse(status="success", data=results)
  except Exception as e:
    logger.exception("Error during similarity search")
    return StandardAPIResponse(status="error", message=str(e))

@router.post("/chat", response_model=StandardAPIResponse)
async def chat(request: ChatRequest):
  try:
    message = request.message
    model_name = request.model_name
    model_provider = request.model_provider.lower()
    logger.debug(f"Chat request for model: {request.model_name} (provider: {request.model_provider})")

    if model_provider not in MODEL_OPTIONS:
      logger.warning("Invalid model provider.")
      return StandardAPIResponse(status="error", message="Invalid model provider.")
    if model_name not in MODEL_OPTIONS[model_provider]["models"]:
      logger.warning("Invalid model name.")
      return StandardAPIResponse(status="error", message="Invalid model name.")

    vectorstore = load_vectorstore(model_provider)
    chain = build_llm_chain(model_provider, model_name, vectorstore,
                            retrieval_mode=request.retrieval_mode)

    if not chain:
      logger.error("Failed to build LLM chain.")
      return StandardAPIResponse(status="error", message="Failed to create LLM chain.")

    result = chain.invoke({"input": message})
    logger.debug("Chat response generated successfully")

    if request.include_context:
      data = {"answer": result["answer"], "context": result.get("context", [])}
    else:
      data = result["answer"]

    return StandardAPIResponse(status="success", data=data)
  except Exception as e:
    logger.exception("Chat endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))
