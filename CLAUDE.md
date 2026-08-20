# CLAUDE.md — Project context for Claude Code

## What this project is

A production-grade RAG system, built by extending the open-source repo
[Zlash65/rag-bot-fastapi](https://github.com/Zlash65/rag-bot-fastapi) (FastAPI + LangChain + ChromaDB +
Streamlit). The base repo does naive top-k dense similarity retrieval only. This project adds, in order:
hybrid retrieval, cross-encoder reranking, a quantified eval harness, GraphRAG knowledge-graph
augmentation, multimodal ingestion, and LangGraph-orchestrated control flow.

**Domain:** research papers, scoped to Retrieval-Augmented Generation / efficient LLM retrieval methods —
~40-50 papers from arXiv (`cs.CL`, `cs.IR`, relevant `cs.LG`), last ~3 years, plus a small deliberately-
seeded set of older foundational papers (DPR, RAG, cross-encoder reranking, GraphRAG itself) so the
citation graph has real internal connectivity rather than mostly pointing outside the corpus. This shapes
several downstream choices: expect benchmark/result tables and architecture-diagram figures (clean,
born-digital, not scanned) driving the Phase 5 multimodal work, and expect the Phase 4 graph schema to
center on entities like paper, author, method, dataset, and benchmark — with a **structured** relation
(`cites`, sourced from the Semantic Scholar Graph API, not LLM-extracted) plus **LLM-extracted** relations
(`evaluates_on`, `extends`, `outperforms`) that only exist in unstructured text. The split matters: pure
citation lookups are answerable by a metadata API alone, so the LLM-extracted layer is what actually
justifies GraphRAG over just calling that API directly.

Full phase breakdown and current status live in `PROJECT_LOG.md` — read that first every session.

## Non-negotiables

- No HlthTek/NHCX code, data, schemas, or business logic anywhere in this repo. Public data only.
- Corpus documents must come from arXiv (public API) and, for citation metadata, the Semantic Scholar
  Graph API / OpenAlex (both public, free, no scraping) — no paywalled sources.
- Every change that touches retrieval must be re-validated against the eval harness (once Phase 1 exists)
  before being considered done. If MRR/nDCG/Recall drops, that's a regression, not a style choice.
- Keep MIT license attribution to the original repo author (Zlash65) in the README credits.
- Don't silently skip a phase's "why" — every phase in PROJECT_LOG.md should end with a one-line
  justification recorded in the Key Architecture Decisions table, because that table is the interview
  cheat sheet this whole project exists to produce.

## Session workflow

1. Read `PROJECT_LOG.md` — check the status board for the current phase and any open TODOs.
2. Do the work for that phase.
3. Before ending the session: update the status board checkboxes, append an entry to the Session Log
   with date / what was done / files touched / eval numbers if they changed / what's next.
4. If a real architectural choice was made (vector DB, embedding model, reranker, fusion method, etc.),
   add a row to the Key Architecture Decisions table with the alternatives considered and why.

## Key file map (base repo structure)

```
rag-bot-v3/
├── pyproject.toml                  # Single source of dependencies for client + server (setuptools backend)
├── client/                         # Streamlit frontend — leave mostly as-is until Phase 7 polish
│   ├── app.py
│   ├── components/{chat,inspector,sidebar}.py
│   ├── state/session.py
│   └── utils/{api,config,helpers}.py
├── server/                         # FastAPI backend — most of the work happens here
│   ├── api/{routes,schemas}.py     # Extend routes.py as new pipeline stages are added
│   ├── core/
│   │   ├── document_processor.py   # PDF validation + chunking → extend here for Phase 5 (multimodal)
│   │   ├── llm_chain_factory.py    # Builds LangChain chains → extend here for Phase 3 (rerank) and
│   │   │                            #   replace/wrap here for Phase 6 (LangGraph StateGraph)
│   │   └── vector_database.py      # Chroma ops → becomes HybridRetriever in Phase 2, gains a graph
│   │                                #   query path in Phase 4
│   ├── config/settings.py          # App config, model provider setup
│   └── main.py
```

Added in Phase 1 (already exists):
- `server/core/chunk_ids.py` — deterministic `{arxiv_id}::p{page}::c{ordinal}` chunk ids, written both
  as Chroma's `ids=` and into `metadata["chunk_id"]`. Also owns `CHUNK_SIZE`/`CHUNK_OVERLAP` and
  `corpus_fingerprint()`. **Phase 2's BM25 index and Phase 3's reranker must use these same ids** —
  RRF and reranking join ranked lists of documents, which requires documents to have names.
- `eval/` — `metrics.py` (pure stdlib, no LangChain/Chroma — keep it that way), `golden_set.py`
  (schema + validation), `retrieval.py` (`RetrieverUnderTest` protocol — **Phase 2/3 add a class here
  and change zero lines of metrics.py**), `judge.py`, `run_eval.py`, `generate_questions.py`,
  `inspect_chunks.py`, `tests/`.

Added in Phase 2 (already exists):
- `server/core/bm25.py` — hand-rolled BM25 (tokenizer + inverted index + scorer + `explain()`),
  **pure stdlib, no LangChain/Chroma — keep it that way**, same constraint as `eval/metrics.py`.
  Owns `TOKENIZATION_VERSION`; the tokenization scheme is a recorded architectural decision, not an
  implementation detail — read its docstring before changing it. Verified against `rank_bm25` via
  `scripts/verify_bm25_against_rank_bm25.py` (that library is NOT a dependency; install temporarily,
  run, uninstall).
- `server/core/hybrid_retriever.py` — dense + BM25 arms, `rrf_fuse`, the per-provider index cache
  (invalidate via `reset_hybrid_cache` after ANY write to a collection), and the LangChain adapter.
  RRF `k=60` is fixed a priori and must not be tuned against the golden set. **Phase 3's reranker
  consumes this module's wide top-50.**
- `eval/compare_runs.py` — paired-bootstrap comparison of two run artifacts + the arm-attribution
  table. Every later phase reports its delta through this, not by eyeballing two run summaries.
- `scripts/rrf_sensitivity_sweep.py` — robustness diagnostic only; its output must stay labelled as
  such and never becomes the reported pipeline.

Retrieval mode is `settings.DEFAULT_RETRIEVAL_MODE` (now `hybrid`), overridable per request via
`retrieval_mode` on `/vector_store/search` and `/chat`.

New modules/services to be added as phases progress (create under `server/core/` unless noted):
- `reranker.py` — BGE-reranker-v2-m3 wrapper, reranking `hybrid_retriever`'s wide top-50 (Phase 3)
- `graph_store.py` — Neo4j connection + citation-edge ingestion (Semantic Scholar/OpenAlex, structured —
  no LLM needed) + `LLMGraphTransformer` ingestion for text-derived relations (`evaluates_on`, `extends`,
  `outperforms`) + Cypher QA chain (Phase 4); schema centers on paper/author/method/dataset/benchmark
  entities — see Domain note above
- `multimodal_ingest.py` — Gemini-based parsing of benchmark/result tables and architecture-diagram
  figures before chunking (Phase 5)
- `graph_pipeline.py` or similar — the LangGraph `StateGraph` replacing the linear chain (Phase 6)
- Docker Compose additions for Neo4j and (if you move off Chroma) the new vector DB (Phase 7)

## Commands

- Install dependencies: `pip install -e .` (from repo root — single `pyproject.toml` covers both client and server)
- Run backend: `cd server && uvicorn main:app --reload`
- Run frontend: `cd client && streamlit run app.py`
- Run tests: `pytest eval/tests/ -q` (pytest is in the `dev` extra: `pip install -e ".[dev]"`)
- Ingest the frozen corpus: `python scripts/ingest_corpus.py --provider groq [--reset]`
- Browse chunks while authoring golden questions: `python eval/inspect_chunks.py --arxiv-id 2004.04906`
  (never type a chunk id by hand — unreachable gold silently depresses every metric)
- Run eval harness: `python eval/run_eval.py --provider groq --retriever {dense,hybrid} --variant <name> --golden-set <path>` —
  prints MRR/Recall@k/Sufficiency@k/nDCG@k stratified by `any_of`/`all_required`, writes artifacts to
  `eval/results/`, and prints the `PROJECT_LOG.md` row to copy in manually (it never edits the log
  itself). Rows are only comparable within a golden-set version — see the comparability rule there.
- Compare two runs: `python eval/compare_runs.py eval/results/<baseline_dir> eval/results/<candidate_dir>`
  — paired-bootstrap deltas with 95% CIs, stratified, plus per-arm gold attribution. A phase's claim
  is the delta and its interval, never the raw number.

## Anticipated interview questions to keep answers current for

Keep short written answers to these somewhere in the repo (e.g. `docs/interview_notes.md`) as they get
answered by real implementation choices, not by memory:
- Why hybrid search instead of dense-only?
- Why a cross-encoder reranker instead of relying on embeddings alone — what's the latency/quality tradeoff?
- Why does GraphRAG not replace vector search — when does each path win?
- Why this vector DB / embedding model / reranker over the alternatives? (pull from the Key Architecture
  Decisions table)
- What would break this system at 10x the corpus size, and what would you change?
- How did you validate the corpus and its graph structure, and what's an example of catching a bad
  inclusion before it mattered? (pull from Key Architecture Decisions — the abstract-vs-title
  curation catch and the rejected fabricated citation)
- Tell me about a time a pipeline silently returned a misleading result. (the "0 internal citation
  edges" figure that turned out to mean "citation lookups didn't run," not "no citations exist" —
  see Key Architecture Decisions)
