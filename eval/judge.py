"""LLM-as-judge for generation quality: groundedness and answer correctness.

Two deliberately separate scores
--------------------------------
`groundedness` is judged against the contexts ACTUALLY PASSED TO THE GENERATOR,
not against the gold chunks. That is what faithfulness means -- a system can be
perfectly faithful to context that retrieval got wrong. Conflating it with
correctness would hide exactly that case, which is the most interesting failure
mode a RAG pipeline has.

`answer_correctness` is judged against the golden set's expected_answer.

Judge/generator separation
--------------------------
The generator is a Groq-hosted model (originally llama-3.3-70b-versatile;
swapped to openai/gpt-oss-120b 2026-08-20 after the Llama models were
decommissioned on this key -- see FALLBACK_MODEL below); the judge is Gemini
gemini-3.5-flash -- a different model family, which is the strongest available
defence against self-preference bias, and at least as capable as the generator
(a weaker judge grading a stronger generator is unreliable).

Gemini quota on this key has been unreliable (a 429 on the embedding endpoint on
2026-08-05, and a sibling chat model already exhausted on 2026-07-26 while
gemini-3.5-flash worked), so:
  - `smoke_test_judge()` must pass before a run commits to it;
  - every verdict records which model produced it;
  - a mid-run fallback is recorded per verdict and reported loudly, never
    silently blended. A judge pool that changed under you halfway through is an
    untraceable confound in any later faithfulness comparison.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

JUDGE_PROVIDER = "gemini"
JUDGE_MODEL = "gemini-3.5-flash"
FALLBACK_PROVIDER = "groq"
# 2026-08-20: llama-3.3-70b-versatile is 404 model_not_found on this key
# (decommissioned, confirmed via client.models.list() + a live invoke()).
# openai/gpt-oss-120b is the nearest available equivalent.
FALLBACK_MODEL = "openai/gpt-oss-120b"

PROMPT_VERSION = "v1"
REQUEST_DELAY_S = 1.0

JUDGE_PROMPT = """You are grading a retrieval-augmented question answering system. \
Be strict and literal.

QUESTION:
{question}

CONTEXT PASSAGES THE SYSTEM WAS GIVEN:
{contexts}

THE SYSTEM'S ANSWER:
{answer}

{expected_block}
Score two SEPARATE things.

1. groundedness (1-5): is every factual claim in the answer supported by the CONTEXT \
PASSAGES above? Judge ONLY against those passages, not against your own knowledge. An \
answer that is factually true in the real world but not supported by these passages is \
NOT grounded.
   5 = every claim traceable to the passages
   3 = mostly supported, some unsupported detail
   1 = largely fabricated or contradicts the passages
   If the answer says only that it does not know, score groundedness 5.

2. answer_correctness (1-5): does the answer actually address the question and match \
the expected answer where one is given?
   5 = fully correct and responsive
   3 = partially correct or incomplete
   1 = wrong or non-responsive
   If no expected answer is given, judge responsiveness and plausibility given the context.

List any claims in the answer that the context does NOT support.

Respond with exactly this JSON and nothing else:
{{"groundedness": <1-5>, "answer_correctness": <1-5>, \
"unsupported_claims": ["<claim>", ...], "rationale": "<two sentences max>"}}"""


@dataclass(frozen=True)
class JudgeVerdict:
  qid: str
  groundedness: int
  answer_correctness: int
  unsupported_claims: Tuple[str, ...]
  rationale: str
  judge_provider: str
  judge_model: str
  prompt_version: str
  cached: bool = False
  raw: str = ""


@dataclass
class Judge:
  """Wraps a primary judge model plus an explicitly-tracked fallback."""
  provider: str = JUDGE_PROVIDER
  model: str = JUDGE_MODEL
  fallback_provider: Optional[str] = FALLBACK_PROVIDER
  fallback_model: Optional[str] = FALLBACK_MODEL
  cache_path: Optional[Path] = None
  _llm: Any = field(default=None, repr=False)
  _fallback_llm: Any = field(default=None, repr=False)
  _cache: Dict[str, dict] = field(default_factory=dict, repr=False)
  fallback_count: int = 0

  def __post_init__(self):
    from core.llm_chain_factory import get_llm
    self._llm = get_llm(self.provider, self.model)
    if self.cache_path and Path(self.cache_path).exists():
      self._cache = json.loads(Path(self.cache_path).read_text(encoding="utf-8"))

  def _get_fallback(self):
    if self._fallback_llm is None and self.fallback_provider:
      from core.llm_chain_factory import get_llm
      self._fallback_llm = get_llm(self.fallback_provider, self.fallback_model)
    return self._fallback_llm

  def save_cache(self):
    if self.cache_path:
      Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
      Path(self.cache_path).write_text(json.dumps(self._cache, indent=2), encoding="utf-8")


def content_to_text(content: Any) -> str:
  """Normalize a LangChain message `.content` to a string.

  ChatGroq returns a plain str, but ChatGoogleGenerativeAI returns a LIST of
  content parts. Assuming str made the Gemini judge fail its smoke test with a
  TypeError that looked, at a glance, exactly like the quota exhaustion this key
  has a history of -- i.e. a code bug wearing the costume of an infra problem.
  """
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for part in content:
      if isinstance(part, str):
        parts.append(part)
      elif isinstance(part, dict):
        parts.append(part.get("text", ""))
    return "".join(parts)
  return str(content)


def _parse(raw: Any) -> Optional[dict]:
  text = content_to_text(raw)
  match = re.search(r"\{.*\}", text, re.S)
  if not match:
    return None
  try:
    return json.loads(match.group(0))
  except json.JSONDecodeError:
    return None


def _cache_key(question: str, answer: str, contexts: Sequence[str]) -> str:
  payload = json.dumps({"q": question, "a": answer, "c": list(contexts),
                        "p": PROMPT_VERSION}, sort_keys=True)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def smoke_test_judge(provider: str = JUDGE_PROVIDER, model: str = JUDGE_MODEL) -> Tuple[bool, str]:
  """One live call before committing a whole run to this model.

  Deliberately not a mere "does it respond" check -- it verifies the model can
  correctly REJECT an ungrounded answer. A judge that scores everything faithful
  would pass a liveness check and still be useless.
  """
  from core.llm_chain_factory import get_llm
  try:
    llm = get_llm(provider, model)
    prompt = JUDGE_PROMPT.format(
      question="What retrieval method does the paper use?",
      contexts="[1] The system uses BM25 sparse retrieval over the corpus.",
      answer="The system uses a graph neural network trained on ImageNet for 400 epochs.",
      expected_block="EXPECTED ANSWER:\nBM25 sparse retrieval.\n\n",
    )
    raw = content_to_text(llm.invoke(prompt).content)
    parsed = _parse(raw)
    if not parsed:
      return False, f"unparseable response: {raw[:200]!r}"
    grounded = int(parsed.get("groundedness", 5))
    if grounded > 2:
      return False, (f"judge scored a clearly fabricated answer groundedness="
                     f"{grounded}; it cannot discriminate")
    return True, f"ok (correctly scored a fabricated answer groundedness={grounded})"
  except Exception as exc:                      # noqa: BLE001
    return False, f"{exc.__class__.__name__}: {exc}"


def judge_answer(judge: Judge, qid: str, question: str, answer: str,
                 contexts: Sequence[str], expected_answer: Optional[str] = None,
                 retries: int = 2) -> JudgeVerdict:
  key = _cache_key(question, answer, contexts)
  if key in judge._cache:
    cached = judge._cache[key]
    return JudgeVerdict(qid=qid, cached=True, **{k: (tuple(v) if k == "unsupported_claims" else v)
                                                 for k, v in cached.items()})

  context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
  expected_block = (f"EXPECTED ANSWER (reference):\n{expected_answer}\n\n"
                    if expected_answer else "")
  prompt = JUDGE_PROMPT.format(question=question, contexts=context_block,
                               answer=answer, expected_block=expected_block)

  for provider, model, llm in (
    (judge.provider, judge.model, judge._llm),
    (judge.fallback_provider, judge.fallback_model, None),
  ):
    if provider is None:
      break
    if llm is None:
      llm = judge._get_fallback()
      if llm is None:
        break
      judge.fallback_count += 1
      print(f"    FALLBACK judge -> {provider}/{model} for {qid}")

    for attempt in range(retries):
      try:
        raw = content_to_text(llm.invoke(prompt).content)
        parsed = _parse(raw)
        if parsed:
          verdict = JudgeVerdict(
            qid=qid,
            groundedness=int(parsed.get("groundedness", 0)),
            answer_correctness=int(parsed.get("answer_correctness", 0)),
            unsupported_claims=tuple(parsed.get("unsupported_claims", []) or []),
            rationale=str(parsed.get("rationale", "")),
            judge_provider=provider,
            judge_model=model,
            prompt_version=PROMPT_VERSION,
            raw=raw[:2000],
          )
          judge._cache[key] = {
            "groundedness": verdict.groundedness,
            "answer_correctness": verdict.answer_correctness,
            "unsupported_claims": list(verdict.unsupported_claims),
            "rationale": verdict.rationale,
            "judge_provider": provider,
            "judge_model": model,
            "prompt_version": PROMPT_VERSION,
            "raw": verdict.raw,
          }
          time.sleep(REQUEST_DELAY_S)
          return verdict
      except Exception as exc:                  # noqa: BLE001
        print(f"    judge error ({exc.__class__.__name__}) attempt {attempt + 1}")
        time.sleep(3 * (attempt + 1))

  raise RuntimeError(f"judge failed for {qid} on both primary and fallback models")


def aggregate_verdicts(verdicts: Sequence[JudgeVerdict]) -> Dict[str, Any]:
  if not verdicts:
    return {}
  n = len(verdicts)
  by_model: Dict[str, int] = {}
  for v in verdicts:
    by_model[f"{v.judge_provider}/{v.judge_model}"] = by_model.get(
      f"{v.judge_provider}/{v.judge_model}", 0) + 1
  return {
    "n": n,
    "mean_groundedness": sum(v.groundedness for v in verdicts) / n,
    "mean_answer_correctness": sum(v.answer_correctness for v in verdicts) / n,
    "pct_fully_grounded": sum(1 for v in verdicts if v.groundedness == 5) / n,
    "pct_ungrounded": sum(1 for v in verdicts if v.groundedness <= 2) / n,
    "judged_by": by_model,
    "judge_pool_blended": len(by_model) > 1,
  }


# --------------------------------------------------------------------------
# judge validation against human labels
# --------------------------------------------------------------------------

def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
  """Cohen's kappa on the paired label sequences (treated as categorical)."""
  if len(a) != len(b) or not a:
    raise ValueError("label sequences must be the same non-zero length")
  n = len(a)
  labels = sorted(set(a) | set(b))
  observed = sum(1 for x, y in zip(a, b) if x == y) / n
  expected = sum((sum(1 for x in a if x == lbl) / n) * (sum(1 for y in b if y == lbl) / n)
                 for lbl in labels)
  if expected == 1.0:
    return 1.0
  return (observed - expected) / (1 - expected)


def validation_report(human: Mapping[str, int], machine: Mapping[str, int],
                      tolerance: int = 1) -> Dict[str, Any]:
  """Agreement between hand-scored labels and judge labels on the same items."""
  shared = sorted(set(human) & set(machine))
  if not shared:
    return {"n": 0, "error": "no overlapping item ids"}
  h = [human[k] for k in shared]
  m = [machine[k] for k in shared]
  exact = sum(1 for x, y in zip(h, m) if x == y) / len(shared)
  within = sum(1 for x, y in zip(h, m) if abs(x - y) <= tolerance) / len(shared)
  mean_err = sum(abs(x - y) for x, y in zip(h, m)) / len(shared)

  # the discriminating check: does the judge REJECT what the human rejected?
  human_bad = [k for k in shared if human[k] <= 2]
  caught = sum(1 for k in human_bad if machine[k] <= 2)
  return {
    "n": len(shared),
    "exact_agreement": exact,
    f"within_{tolerance}_agreement": within,
    "mean_absolute_error": mean_err,
    "cohens_kappa": cohens_kappa(h, m),
    "n_human_rejected": len(human_bad),
    "n_judge_also_rejected": caught,
    "rejection_recall": (caught / len(human_bad)) if human_bad else None,
    "disagreements": [{"id": k, "human": human[k], "judge": machine[k]}
                      for k in shared if abs(human[k] - machine[k]) > tolerance],
  }
