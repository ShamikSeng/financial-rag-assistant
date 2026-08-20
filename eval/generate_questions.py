"""Draft the bulk half of the golden set: stratified sampling + LLM question generation.

This produces the ~75 LLM-generated questions. The ~20 hand-written hard
questions (paraphrase / cross-paper / multi-hop / table-dependent) are authored
separately by hand and merged in -- deliberately, because a question set written
entirely by a model that was shown the answer chunk inherits that model's idea
of what is hard, which is exactly the bias the hard set exists to counter.

Sampling
--------
The FULL plan (~75) is drawn up front with a fixed seed, and the pilot is its
first N. So the pilot is a genuine sample of the final design rather than a
hand-picked easy subset -- "the harness works on the pilot" then actually
derisks the full run.

Allocation is a floor of >=1 chunk per paper (all 45 papers represented) with
the remainder distributed proportional to chunk count, since counts are heavily
skewed (140 max vs ~47 mean) and pure proportional sampling would let a few long
papers dominate.

Leakage
-------
An LLM shown a chunk tends to write a question reusing its distinctive
vocabulary, which dense retrieval then finds trivially -- inflating the Phase 1
baseline and shrinking the measurable lift from Phases 2-3. The prompt pushes
against this, and then `question_chunk_jaccard` MEASURES the residual rather
than assuming the prompt worked. Measuring it is the defensible move; asserting
that a prompt fixed it is not.

Usage (from repo root):
    python eval/generate_questions.py --plan-only
    python eval/generate_questions.py --limit 18 --out eval/data/pilot_draft.json
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._bootstrap import REPO_ROOT, bootstrap_server_context  # noqa: E402

bootstrap_server_context()

from core.chunk_ids import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_VERSION, corpus_fingerprint, parse_chunk_id  # noqa: E402
from eval.golden_set import content_words, question_chunk_jaccard  # noqa: E402
from eval.retrieval import load_collection_index  # noqa: E402

# llama-3.3-70b-versatile hit its daily TPD cap (100,000 tokens; a real generation
# run needs ~4K tokens/accepted question) mid-session on 2026-08-17 with zero
# questions produced. Switched to llama-3.1-8b-instant: a SEPARATE Groq quota
# bucket, so it doesn't compete with anything else on this key, and it keeps
# Gemini purely in the judge role rather than also authoring expected_answer
# (which would create a mild circularity against later answer_correctness
# judging). Weaker model, but the self-verification + named-entity + dedup
# filters added this session are the real quality gate now, not raw model
# strength -- see the Key Architecture Decisions rows in PROJECT_LOG.md.
#
# 2026-08-20: both llama-3.1-8b-instant and llama-3.3-70b-versatile were found
# 404 model_not_found on this key (decommissioned, not a quota issue this
# time -- confirmed via a live client.models.list() call). Swapped to the
# nearest available equivalent; re-verify with a real invoke() before trusting
# any future swap, same lesson as the two prior stale-model incidents in
# PROJECT_LOG.md's session log.
GENERATOR_MODEL = "openai/gpt-oss-20b"
GENERATOR_PROVIDER = "groq"

TOTAL_QUESTIONS = 75
SEED = 20260814
MIN_CHUNK_CHARS = 700          # skip fragments: headers, stub pages, caption-only chunks
REQUEST_DELAY_S = 1.5          # unauthenticated-ish groq tier; be polite
DUPLICATE_JACCARD = 0.35       # question-vs-question; above this they compete for the same chunks


def question_pair_jaccard(a: str, b: str) -> float:
  wa, wb = content_words(a), content_words(b)
  if not wa or not wb:
    return 0.0
  return len(wa & wb) / len(wa | wb)

MANIFEST_PATH = REPO_ROOT / "data" / "papers" / "corpus_manifest.json"

# Chunks that are bibliography or bare tables make terrible question sources --
# a question about a reference list tests nothing about retrieval quality. A
# first pass with a hand-guessed filter had the model rejecting 12 of 18 sampled
# chunks, so these thresholds were instead fitted to those measured examples
# (see the feature table in the session log): reference lists show 5-8
# venue-phrase hits, tables show alpha_ratio < 0.68 or high digit density with
# almost no sentence structure.
_VENUE_RE = re.compile(
  r"In Proceedings|Proceedings of|Conference on|Transactions on|Journal of|preprint|"
  r"arXiv preprint|pages \d+", re.I)
_ARXIV_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_BRACKET_REF_RE = re.compile(r"\[\d+\]")
# prose marker: lowercase word, period, space, capital -- i.e. a sentence boundary
_SENTENCE_RE = re.compile(r"[a-z]{3,}\.\s+[A-Z]")


def chunk_quality(text: str) -> Tuple[bool, str]:
  """(is_usable, reason). Cheap lexical screen before spending an LLM call."""
  n = max(len(text), 1)
  alpha_ratio = sum(c.isalpha() for c in text) / n
  digit_ratio = sum(c.isdigit() for c in text) / n
  venues = len(_VENUE_RE.findall(text))
  years = len(_YEAR_RE.findall(text))
  arxiv = len(_ARXIV_RE.findall(text))
  brackets = len(_BRACKET_REF_RE.findall(text))
  sentences = len(_SENTENCE_RE.findall(text))

  if len(text) < MIN_CHUNK_CHARS:
    return False, f"too short ({len(text)} chars)"
  if venues >= 3:
    return False, f"bibliography ({venues} venue phrases)"
  if years >= 5 and venues >= 1:
    return False, f"bibliography ({years} years, {venues} venue phrases)"
  if arxiv >= 2 and years >= 4:
    return False, f"bibliography ({arxiv} arXiv ids, {years} years)"
  if brackets >= 4 and venues >= 1:
    return False, f"bibliography ({brackets} bracketed refs)"
  if alpha_ratio < 0.68:
    return False, f"table/numeric (alpha_ratio {alpha_ratio:.2f})"
  if digit_ratio > 0.07 and sentences <= 3:
    return False, f"table (digit_ratio {digit_ratio:.2f}, {sentences} sentences)"
  if sentences <= 1:
    return False, "no sentence structure"
  return True, "ok"


GEN_PROMPT = """You are helping build an evaluation set for a retrieval system over \
research papers about Retrieval-Augmented Generation and neural retrieval.

Below is ONE passage from a paper. Write ONE question that this passage answers.

Hard requirements:
1. PARAPHRASE. Do NOT reuse the passage's distinctive phrasing. If the passage says \
"in-batch negatives", ask about "how negative examples are chosen during training". \
A question that copies the passage's wording is useless to us -- it tests string \
matching, not retrieval.
2. NAME SOMETHING SPECIFIC. The question MUST name the particular method, system, \
model, dataset, or benchmark it is about. "How does TabRank use intermediate reasoning \
signals?" is good. "How are negative examples chosen in RAG systems?" is BAD -- dozens \
of papers answer that, so it identifies no single passage.
3. SELF-CONTAINED. Never write "this paper", "the authors", "the above method", \
"Table 2". Someone who has not seen the passage must understand it.
4. ANSWERABLE FROM THIS PASSAGE ALONE. If the passage only mentions a topic in \
passing, or cites other work about it, that is NOT enough to ask about.
5. NATURAL. Write it the way a researcher searching for this information would type it.

If the passage is NOT suitable -- a reference list, author list, page header, fragment, \
table of contents, or no substantive claim of its own -- respond with exactly: \
{{"skip": true, "reason": "<short reason>"}}

Otherwise respond with exactly this JSON and nothing else:
{{"skip": false, "question": "<your question>", "entity": "<the exact method/system/dataset \
name you used, copied verbatim from your question>", "answer": "<concise answer, 1-3 \
sentences, grounded only in the passage>"}}

PASSAGE:
\"\"\"
{chunk_text}
\"\"\""""

# Applied to the SOURCE chunk as well as to adjacent candidates. The pilot run
# showed why: the generator happily invented questions from passages that only
# mention a topic in passing (a related-work citation list produced "how are
# negative examples chosen during training"), and a title page produced a
# question about how the system works. Those questions are unanswerable from
# their own gold chunk, so they score as permanent misses no matter how good
# retrieval gets -- poisoning the baseline AND capping every later phase.
#
# The bug was an asymmetry: neighbours had to pass a strict admission test while
# the source chunk was assumed to be gold by construction. Now both face it.
ANSWERS_PROMPT = """A retrieval evaluation set needs to know whether a passage \
independently answers a question.

QUESTION: {question}

PASSAGE:
\"\"\"
{chunk_text}
\"\"\"

Does this passage ALONE contain enough information to answer the question fully? \
Be strict. All of these count as NO:
- the passage only mentions the topic in passing
- the passage cites other work about the topic without explaining it
- the passage is a title, author list, or reference list that merely contains the term
- the passage gives related background but not the specific answer

Respond with exactly this JSON and nothing else:
{{"answers": true|false, "reason": "<short reason>"}}"""


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def allocate(paper_chunk_counts: Dict[str, int], total: int) -> Dict[str, int]:
  """>=1 per paper, remainder proportional to chunk count (largest remainder)."""
  papers = sorted(paper_chunk_counts)
  n_papers = len(papers)
  if total < n_papers:
    raise ValueError(f"total ({total}) < number of papers ({n_papers}); every paper needs >= 1")

  alloc = {p: 1 for p in papers}
  remaining = total - n_papers
  if remaining <= 0:
    return alloc

  grand_total = sum(paper_chunk_counts.values())
  exact = {p: remaining * paper_chunk_counts[p] / grand_total for p in papers}
  for p in papers:
    alloc[p] += int(exact[p])

  # largest-remainder to distribute the rounding slack deterministically
  slack = total - sum(alloc.values())
  by_remainder = sorted(papers, key=lambda p: (-(exact[p] - int(exact[p])), p))
  for p in by_remainder[:slack]:
    alloc[p] += 1
  return alloc


def build_pools(chunk_ids_by_paper: Dict[str, List[str]], text_by_id: Dict[str, str],
                seed: int = SEED) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
  """Per-paper pools of usable chunks, deterministically shuffled.

  Returns (pools, rejection_reasons). Pools are longer than the paper's quota on
  purpose: the model still declines some chunks the lexical screen lets through
  (a passage can be clean prose and still carry no answerable claim), and when
  that happens generation draws the next pool entry for the SAME paper rather
  than losing the slot. Without that, per-paper stratification silently decays
  into "whichever papers happened to have quotable chunks".
  """
  rng = random.Random(seed)
  pools: Dict[str, List[str]] = {}
  reasons: Dict[str, str] = {}

  for paper in sorted(chunk_ids_by_paper):
    usable = []
    for cid in sorted(chunk_ids_by_paper[paper]):
      ok, reason = chunk_quality(text_by_id[cid])
      if ok:
        usable.append(cid)
      else:
        reasons[cid] = reason
    if not usable:                       # never let a paper drop out entirely
      usable = sorted(chunk_ids_by_paper[paper])
    rng.shuffle(usable)
    pools[paper] = usable

  return pools, reasons


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

def build_llm():
  from core.llm_chain_factory import get_llm
  return get_llm(GENERATOR_PROVIDER, GENERATOR_MODEL)


def parse_json_response(raw: str) -> Optional[dict]:
  match = re.search(r"\{.*\}", raw, re.S)
  if not match:
    return None
  try:
    return json.loads(match.group(0))
  except json.JSONDecodeError:
    return None


class QuotaExhaustedError(RuntimeError):
  """A DAILY/long-window limit, distinct from a transient per-minute 429.

  The distinction matters: a per-minute limit clears in seconds, so a short
  backoff is correct. A daily-cap 429 will not clear no matter how many times a
  chunk is retried. The 2026-08-17 run burned 17 minutes and produced zero
  questions doing exactly that -- flat 5/10/15s backoff per chunk, then straight
  on to the next chunk with no pause between chunks at all, so the effective
  call cadence never gave the daily bucket any room to recover.
  """


def is_daily_quota_error(exc: Exception) -> bool:
  text = str(exc).lower()
  return "tokens per day" in text or "requests per day" in text or " tpd" in text


def call_llm(llm, prompt: str, retries: int = 3) -> Optional[dict]:
  for attempt in range(retries):
    try:
      raw = llm.invoke(prompt).content
      parsed = parse_json_response(raw)
      if parsed is not None:
        return parsed
      # one repair attempt before giving up on this chunk
      prompt = prompt + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
    except Exception as exc:            # noqa: BLE001 -- rate limits, transient API errors
      if is_daily_quota_error(exc):
        # Do not retry at all -- retrying a daily-cap error just spends the
        # attempt budget on a wait it cannot outlast. Surface it to the caller,
        # which aborts the whole run rather than silently grinding through
        # every remaining chunk at zero throughput.
        raise QuotaExhaustedError(str(exc)) from exc
      wait = 5 * (attempt + 1)
      print(f"    LLM error ({exc.__class__.__name__}); retrying in {wait}s", file=sys.stderr)
      time.sleep(wait)
  return None


def neighbours_of(chunk_id: str, known: set) -> List[str]:
  """Adjacent chunks on the same page -- the ones that actually share overlap.

  TokenTextSplitter splits each page-Document independently, so the 50-token
  overlap only ever exists WITHIN a page. Chunks on different pages never
  overlap, which is why this does not look across page boundaries.
  """
  p = parse_chunk_id(chunk_id)
  out = []
  for delta in (-1, 1):
    if p.ordinal + delta < 0:
      continue
    candidate = f"{p.arxiv_id}::p{p.page}::c{p.ordinal + delta}"
    if candidate in known:
      out.append(candidate)
  return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--provider", default="groq")
  ap.add_argument("--total", type=int, default=TOTAL_QUESTIONS,
                  help="per-paper quota is computed against this full-set size")
  ap.add_argument("--target", type=int, help="stop after this many accepted questions")
  ap.add_argument("--seed", type=int, default=SEED)
  ap.add_argument("--plan-only", action="store_true", help="show the allocation and exit")
  ap.add_argument("--no-neighbours", action="store_true",
                  help="skip the adjacent-chunk equivalence check (halves LLM calls)")
  ap.add_argument("--no-specificity", action="store_true",
                  help="drop the named-entity requirement. Only for authoring the "
                       "hand-written hard set, which is deliberately exempt -- a good "
                       "paraphrase question avoids naming the method by its exact term.")
  ap.add_argument("--exclude", help="path to an existing draft whose source chunks to skip "
                                    "(use when extending the pilot to the full set)")
  ap.add_argument("--qid-prefix", default="b")
  ap.add_argument("--out", default="eval/data/bulk_draft.json")
  args = ap.parse_args(argv)

  ids, sha_by_id, text_by_id = load_collection_index(args.provider)

  by_paper: Dict[str, List[str]] = defaultdict(list)
  for cid in ids:
    by_paper[parse_chunk_id(cid).arxiv_id].append(cid)

  pools, rejection_reasons = build_pools(by_paper, text_by_id, seed=args.seed)
  quota = allocate({p: len(c) for p, c in by_paper.items()}, args.total)

  already_used = set()
  if args.exclude:
    prior = json.loads((REPO_ROOT / args.exclude).read_text(encoding="utf-8"))
    for q in prior.get("questions", []):
      already_used.update(q["gold_groups"][0]["chunk_ids"])
    already_used.update(c for c, _ in prior.get("generation", {}).get("skipped", []))
    print(f"Excluding {len(already_used)} chunks already used by {args.exclude}\n")

  if args.plan_only:
    total_usable = sum(len(p) for p in pools.values())
    print(f"Quota: {sum(quota.values())} questions across {len(quota)} papers (seed={args.seed})")
    print(f"Usable chunks after lexical screen: {total_usable} of {len(ids)} "
          f"({len(rejection_reasons)} rejected)\n")
    reason_counts = Counter(r.split(" (")[0] for r in rejection_reasons.values())
    for reason, n in reason_counts.most_common():
      print(f"  rejected: {reason:<28} {n}")
    print()
    for paper, n in sorted(quota.items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
      print(f"  {paper:>14}  quota {n}  pool {len(pools[paper])}  (of {len(by_paper[paper])} chunks)")
    thin = [p for p in pools if len(pools[p]) < quota[p]]
    if thin:
      print(f"\n  WARNING: {len(thin)} papers have a pool smaller than their quota: {thin}")
    return 0

  target = args.target or sum(quota.values())
  llm = build_llm()
  print(f"Target {target} questions, quota across {len(quota)} papers, model {GENERATOR_MODEL}\n")

  papers = sorted(pools)
  random.Random(args.seed).shuffle(papers)
  cursor = {p: 0 for p in papers}
  accepted_per_paper: Counter = Counter()

  questions = []
  skipped = []
  attempt = 0

  quota_exhausted_msg = None

  def safe_call(prompt: str) -> Optional[dict]:
    """call_llm, but a daily-quota error is recorded and re-raised as the
    sentinel that stops the round-robin loop below, instead of being retried
    or treated as an ordinary per-chunk failure."""
    nonlocal quota_exhausted_msg
    try:
      return call_llm(llm, prompt)
    except QuotaExhaustedError as exc:
      quota_exhausted_msg = str(exc)
      raise

  # Round-robin passes over papers. A model skip advances only that paper's
  # cursor, so the slot is refilled from the same paper instead of being lost.
  while len(questions) < target:
    progressed = False
    for paper in papers:
      if len(questions) >= target:
        break
      if accepted_per_paper[paper] >= quota[paper]:
        continue
      while cursor[paper] < len(pools[paper]) and pools[paper][cursor[paper]] in already_used:
        cursor[paper] += 1
      if cursor[paper] >= len(pools[paper]):
        continue

      chunk_id = pools[paper][cursor[paper]]
      cursor[paper] += 1
      progressed = True
      attempt += 1

      print(f"[{len(questions) + 1}/{target}] (try {attempt}) {chunk_id}", flush=True)
      try:
        result = safe_call(GEN_PROMPT.format(chunk_text=text_by_id[chunk_id]))
      except QuotaExhaustedError:
        break
      time.sleep(REQUEST_DELAY_S)

      if result is None:
        skipped.append((chunk_id, "llm_failed"))
        print("    SKIP (no parseable response)")
        continue
      if result.get("skip"):
        skipped.append((chunk_id, result.get("reason", "model skipped")))
        print(f"    SKIP ({result.get('reason', '')})")
        continue

      question = (result.get("question") or "").strip()
      answer = (result.get("answer") or "").strip()
      entity = (result.get("entity") or "").strip()
      if not question:
        skipped.append((chunk_id, "empty question"))
        continue

      # --- specificity: the named entity must actually appear in the question --
      # Scoped to bulk generation ONLY. Hand-written hard questions are exempt:
      # a good paraphrase question deliberately avoids naming the method by its
      # exact term, and forcing an entity into it would defeat the reason that
      # question exists.
      if not args.no_specificity:
        if not entity or entity.lower() not in question.lower():
          skipped.append((chunk_id, f"no specific named entity (entity={entity!r})"))
          print(f"    SKIP (underspecified -- entity {entity!r} not in question)")
          continue

      # --- near-duplicate rejection ----------------------------------------
      # Two questions asking the same thing with different gold cannot both
      # score: retrieval returns the same chunks for both, so one is a
      # guaranteed miss that says nothing about retrieval quality.
      dupe = None
      for prior in questions:
        pj = question_pair_jaccard(question, prior["question"])
        if pj >= DUPLICATE_JACCARD:
          dupe = (prior["qid"], pj)
          break
      if dupe:
        skipped.append((chunk_id, f"near-duplicate of {dupe[0]} (jaccard {dupe[1]:.2f})"))
        print(f"    SKIP (near-duplicate of {dupe[0]}, jaccard={dupe[1]:.2f})")
        continue

      # --- the source chunk must pass the SAME test as any candidate --------
      try:
        self_check = safe_call(ANSWERS_PROMPT.format(question=question, chunk_text=text_by_id[chunk_id]))
      except QuotaExhaustedError:
        break
      time.sleep(REQUEST_DELAY_S)
      if not self_check or self_check.get("answers") is not True:
        reason = (self_check or {}).get("reason", "self-verification failed")
        skipped.append((chunk_id, f"source chunk does not answer its own question: {reason}"))
        print(f"    SKIP (gold would be wrong -- {reason})")
        continue

      jaccard = question_chunk_jaccard(question, text_by_id[chunk_id])

      group_ids = [chunk_id]
      neighbour_notes = []
      quota_hit_in_neighbours = False
      if not args.no_neighbours:
        for neighbour in neighbours_of(chunk_id, ids):
          try:
            verdict = safe_call(ANSWERS_PROMPT.format(question=question, chunk_text=text_by_id[neighbour]))
          except QuotaExhaustedError:
            quota_hit_in_neighbours = True
            break
          time.sleep(REQUEST_DELAY_S)
          if verdict and verdict.get("answers") is True:
            group_ids.append(neighbour)
            neighbour_notes.append(f"{neighbour}: {verdict.get('reason', '')}")
            print(f"    + equivalent: {neighbour}")
      if quota_hit_in_neighbours:
        # This question's own gold (group_ids[0]) is already verified -- keep
        # it rather than discarding a fully-valid question just because its
        # neighbour check didn't finish. Any_of with a single confirmed member
        # is still a fully valid gold group.
        pass

      accepted_per_paper[paper] += 1
      questions.append({
        "qid": f"{args.qid_prefix}{len(questions) + 1:04d}",
        "question": question,
        "semantics": "any_of",
        "gold_groups": [{
          "chunk_ids": sorted(group_ids),
          "note": "source chunk" + (f"; adjacent equivalents verified -- {'; '.join(neighbour_notes)}"
                                    if neighbour_notes else ""),
        }],
        "expected_answer": answer,
        "source_papers": [parse_chunk_id(chunk_id).arxiv_id],
        "provenance": "llm_generated",
        "tags": ["bulk"],
        "chunk_text_sha8": {cid: sha_by_id[cid] for cid in sorted(group_ids)},
        "_review": {
          "source_chunk": chunk_id,
          "question_chunk_jaccard": round(jaccard, 4),
          "leakage_flag": jaccard >= 0.35,
        },
      })
      flag = "  <-- HIGH LEAKAGE" if jaccard >= 0.35 else ""
      print(f"    jaccard={jaccard:.3f}{flag}\n    Q: {question}")

      if quota_hit_in_neighbours:
        break        # question is saved above; stop before spending more budget

    if quota_exhausted_msg:
      break
    if not progressed:
      print(f"\nPools exhausted at {len(questions)} questions.")
      break

  if quota_exhausted_msg:
    print(f"\n{'=' * 70}\nSTOPPED: daily quota exhausted after {len(questions)} question(s) "
          f"({attempt} attempts).\n{quota_exhausted_msg}\n{'=' * 70}", file=sys.stderr)

  payload = {
    "version": "draft",
    "corpus_fingerprint": corpus_fingerprint(MANIFEST_PATH),
    "chunking": {
      "splitter": "TokenTextSplitter",
      "chunk_size": CHUNK_SIZE,
      "chunk_overlap": CHUNK_OVERLAP,
      "chunking_version": CHUNKING_VERSION,
    },
    "generation": {
      "model": GENERATOR_MODEL,
      "provider": GENERATOR_PROVIDER,
      "seed": args.seed,
      "plan_total": args.total,
      "attempts": attempt,
      "skipped": skipped,
      "stopped_on_quota_exhaustion": bool(quota_exhausted_msg),
      "quota_error": quota_exhausted_msg,
    },
    "questions": questions,
  }

  out_path = REPO_ROOT / args.out
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  jaccards = [q["_review"]["question_chunk_jaccard"] for q in questions]
  flagged = sum(1 for j in jaccards if j >= 0.35)
  print(f"\n=== {len(questions)} questions, {len(skipped)} source chunks skipped")
  if jaccards:
    ordered = sorted(jaccards)
    print(f"    leakage jaccard: min {ordered[0]:.3f}  median {ordered[len(ordered) // 2]:.3f}  "
          f"max {ordered[-1]:.3f}  |  {flagged} flagged >= 0.35")
  print(f"    wrote {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
