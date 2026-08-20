from config.settings import GROQ_API_KEY, GOOGLE_API_KEY

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.logger import logger


def get_prompt():
  logger.debug("Creating chat prompt template.")
  return ChatPromptTemplate.from_messages([
    ("system", "Answer as detailed as possible using the context below. If unknown, say 'I don't know.'"),
    ("human", "Context:\n{context}\n\n\nQuestion:\n{input}")
  ])

def get_llm(model_provider: str, model: str):
  logger.debug(f"Initializing LLM for {model_provider} - {model}")
  if model_provider == "groq":
    return ChatGroq(model=model, api_key=GROQ_API_KEY)
  elif model_provider == "gemini":
    return ChatGoogleGenerativeAI(model=model, api_key=GOOGLE_API_KEY)
  else:
    logger.error(f"Unsupported LLM Provider: {model_provider}")
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

def build_llm_chain(model_provider: str, model: str, vectorstore, retrieval_mode: str = None):
  """Retrieve -> stuff -> generate.

  Phase 2 note: CLAUDE.md's extension-point table predicted this file would
  first be touched in Phase 3 (rerank). It moves a phase earlier because wiring
  hybrid retrieval into the live app means the chat path needs a retriever that
  is not `vectorstore.as_retriever(...)`, and leaving /chat on dense-only while
  claiming the phase shipped would make the capability eval-harness-only.

  k stays at 3 in both modes. The k=3 (chat) vs k=4 (search endpoint) mismatch
  is still the separately-deferred measured change it was in Phase 1 -- folding
  it in here would confound it with the dense-vs-hybrid delta.
  """
  from core.vector_database import resolve_retrieval_mode

  mode = resolve_retrieval_mode(retrieval_mode)
  logger.debug(f"Building LLM chain for provider: {model_provider}, model: {model}, mode: {mode}")
  prompt = get_prompt()
  llm = get_llm(model_provider, model)

  if mode == "hybrid":
    from core.hybrid_retriever import build_langchain_retriever
    retriever = build_langchain_retriever(model_provider, k=3)
  else:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

  return create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt=prompt)
  )
