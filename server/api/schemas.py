from pydantic import BaseModel
from typing import Any, Optional, Literal


class SearchQueryRequest(BaseModel):
    model_provider: str
    query: str
    # Both optional with behaviour-preserving defaults: omitting them reproduces
    # the pre-Phase-1 response exactly, so the Streamlit client is unaffected.
    k: Optional[int] = None
    include_scores: bool = False
    # Phase 2. None -> settings.DEFAULT_RETRIEVAL_MODE, so existing callers are
    # unaffected either way; "dense" | "hybrid" pins one path explicitly, which
    # is what makes an A/B comparison possible over HTTP without a redeploy.
    retrieval_mode: Optional[Literal["dense", "hybrid"]] = None

class ChatRequest(BaseModel):
    model_provider: str
    model_name: str
    message: str
    # When False (default) `data` stays a bare answer string, which is what
    # client/utils/api.py::chat() renders. When True, `data` becomes
    # {"answer": str, "context": [Document]}.
    #
    # This is Phase-6 motivated, not needed by Phase 1 -- the eval harness
    # retrieves in-process rather than over HTTP. It is added now because it is
    # a small change that makes groundedness measurable through the API, and
    # Phase 6's StateGraph needs the retrieved context exposed anyway. Recorded
    # explicitly per CLAUDE.md's rule that every change carries a justification,
    # rather than letting a Phase-6 change pass quietly inside a Phase-1 session.
    include_context: bool = False

    # Phase 2, same contract as on SearchQueryRequest above.
    retrieval_mode: Optional[Literal["dense", "hybrid"]] = None

class StandardAPIResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[Any] = None
    message: Optional[str] = None
