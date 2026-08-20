# PROJECT_LOG.md

## Project overview

**Name:** TBD (name it once the corpus is frozen)
**Base repo:** [Zlash65/rag-bot-fastapi](https://github.com/Zlash65/rag-bot-fastapi) — cloned on: _fill in date_
**Goal:** Extend a modest FastAPI + LangChain + ChromaDB RAG chatbot into a production-grade retrieval system with hybrid search, cross-encoder reranking, a quantified eval harness (MRR / Recall@k / nDCG@k), GraphRAG knowledge-graph augmentation, multimodal ingestion, and LangGraph-orchestrated control flow.

**Domain:** Research papers on Retrieval-Augmented Generation / efficient LLM retrieval methods. Citation edges come from the Semantic Scholar Graph API (structured, not LLM-extracted); `evaluates_on`/`extends`/`outperforms` relations are LLM-extracted from paper text, which is what actually justifies GraphRAG over just calling a citation API directly.

**Corpus: FROZEN 2026-08-05.** 45 papers (14 anchor/foundational + 31 from an arXiv `cs.CL`/`cs.IR`/`cs.LG` keyword search, last ~3 years), 112 internal citation edges, 89.0 MB of PDFs. Full provenance (source URL, sha256, download date) in `data/papers/corpus_manifest.json`. No additions/removals from here without re-running the Phase 1 eval baseline (see the eval-staging non-negotiable below) — this is the actual corpus Phase 1 onward is measured against.

**IP note:** Personal project. Public data only. No HlthTek/NHCX code, schemas, or documents anywhere in this repo.

**Original base repo stack (as of clone):**
- FastAPI backend (`server/`) + Streamlit frontend (`client/`)
- LangChain retrieval chain, ChromaDB vector store
- Groq + Gemini as pluggable LLM providers
- PyPDF for parsing, `TokenTextSplitter` for chunking
- Retrieval: naive top-k dense similarity search only — no hybrid search, no reranking, no eval harness, no knowledge graph, no multimodal ingestion, no LangGraph

---

## Status board

- [x] Phase 0 — Understand & baseline existing codebase, pick domain, freeze corpus (45 papers, see Project Overview)
- [x] Phase 1 — Eval harness (MRR / Recall@k / nDCG@k) + baseline numbers
- [x] Phase 2 — Hybrid retrieval (dense + BM25 + RRF)
- [ ] Phase 3 — Cross-encoder reranking (BGE-reranker-v2-m3)
- [ ] Phase 4 — GraphRAG (Neo4j + LLMGraphTransformer + query router)
- [ ] Phase 5 — Multimodal ingestion (Gemini parsing for tables/scans)
- [ ] Phase 6 — LangGraph orchestration (explicit StateGraph)
- [ ] Phase 7 — Serving, observability (Prometheus), Docker Compose, polish
- [ ] Phase 8 — Interview packaging (eval table, diagram, Q&A prep)

---

## Phase breakdown

### Phase 0 — Understand & baseline
- [x] Clone repo, get backend (`uvicorn`) and frontend (`streamlit`) running locally
- [x] Trace one full request end-to-end: PDF upload → `document_processor.py` → chunk → embed → `vector_database.py` → Chroma insert
- [x] Trace one full query: question → retriever → `llm_chain_factory.py` → LLM → response
- [x] Write a short architecture summary below (in your own words — this is your interview answer for "walk me through the base system")
- [x] Pick the domain — research papers on RAG/retrieval methods (see Project Overview)
- [x] Shortlist/freeze the actual paper corpus — 45 papers frozen, `data/papers/corpus_manifest.json` (see Project Overview and Corpus source & scope)
- [x] Note extension points (which files change in which phase — see CLAUDE.md)

**Architecture summary (fill in after Phase 0):**
> **Upload flow (PDF → Chroma):** `POST /upload_and_process_pdfs` (`routes.py`) takes a
> `model_provider` + files and hands off to `upsert_vectorstore_from_pdfs`. `document_processor.py`
> validates each file is a `.pdf` under 200MB, saves it to `./temp/uploaded_files/`, loads it with
> `PyPDFLoader` (one `Document` per page), then splits into ~500-token chunks (50-token overlap) via
> `TokenTextSplitter`. `vector_database.py` picks an embedding model per provider — local
> `sentence-transformers/all-MiniLM-L12-v2` for "groq", Gemini's `gemini-embedding-001` API for "gemini" (Groq
> has no embeddings API of its own, so "groq" really means "HF local embeddings + Groq chat model") —
> and either appends to or creates a persisted Chroma collection at `./data/{provider}_vector_store`.
> Each provider's collection is fully separate since embedding spaces aren't cross-compatible.
>
> **Query flow (question → response):** `POST /chat` (`routes.py`) validates the requested
> provider/model against `MODEL_OPTIONS`, reloads the matching persisted Chroma store (same embedding
> function as at insert time, so query and chunk vectors share a space), and calls
> `build_llm_chain` in `llm_chain_factory.py`. That wraps Chroma as a naive top-3 dense-similarity
> retriever (`as_retriever(search_kwargs={"k": 3})` — no hybrid search, no reranking, the exact
> baseline this project's phases replace) and combines it with a "stuff" documents chain: a fixed
> system prompt + all 3 retrieved chunks concatenated directly into context, sent to `ChatGroq` or
> `ChatGoogleGenerativeAI`. `create_retrieval_chain` glues retrieve → stuff → generate into one
> `.invoke()` call; the route pulls out just `["answer"]` and returns it.
>
> **Net takeaway:** the base system is a clean but minimal linear pipeline — one retrieval call,
> one LLM call, no reranking, no query rewriting, no multi-hop reasoning, and no structured handling
> of tables (PyPDF just flattens everything to text). Every later phase in this project targets one
> of these gaps directly: Phase 2 replaces the naive top-k retriever with hybrid dense+BM25, Phase 3
> adds a reranking step between retrieve and stuff, Phase 4 adds a second (graph) retrieval path for
> multi-hop/relational questions this retriever structurally can't answer, Phase 5 fixes the
> table-flattening problem, and Phase 6 turns this implicit two-step chain into an explicit,
> inspectable `StateGraph`.

### Phase 1 — Eval harness
- [x] Ingest the frozen 45-paper corpus into the vector store (prerequisite for everything else in this
  phase — see `scripts/ingest_corpus.py` and the 2026-08-05 (5) session log entry). **groq (local
  HuggingFace embeddings) done: 2131 chunks in `server/data/groq_vector_store`. gemini not yet
  ingested** — blocked on API quota, see Open questions/TODOs.
- [x] **Deterministic chunk IDs + re-ingest** (unplanned prerequisite discovered 2026-08-14 — chunks
  had no stable identity at all, so `expected_chunk_ids` had no referent. See `server/core/chunk_ids.py`
  and the Key Architecture Decisions row.) 2131 chunks re-ingested, ids verified to survive retrieval.
- [x] Implement MRR, Recall@k, nDCG@k against current retriever — `eval/metrics.py`, 60 passing tests
- [x] Eval harness plumbing: `eval/{_bootstrap,golden_set,retrieval,report,run_eval,inspect_chunks}.py`,
  runs end-to-end against the live collection
- [x] Implement a groundedness/faithfulness check — `eval/judge.py` built; `gemini-3.5-flash` smoke-tested
  live and confirmed it correctly *rejects* a fabricated answer (groundedness=1)
- [x] Build 75-100 question golden set — **`eval/data/golden_set.v1.json`, 89 questions frozen
  2026-08-20** (69 bulk `any_of` + 20 handwritten, 10 of them `all_required`). Bulk questions were
  LLM-generated then audited twice — a full-text re-read of every miss/sample/multi-chunk group,
  then an independent second pass by a different model (Gemini) with every one of its flags
  re-verified against source text before any fix was trusted (9/12 flags confirmed real, 2 rejected
  as over-strict — see the 2026-08-20 session log entry and the Key Architecture Decisions row on
  golden-set review methodology). 75 → 71 → 69 across the two rounds, all drops/repoints logged in
  `eval/data/bulk_questions_review.md`.
- [x] Judge validation: ~15-20 answers hand-scored against the LLM judge (blind, shuffled, including
  deliberately unfaithful answers) — **done 2026-08-20.** 20-item bundle (15 clean + 5 deliberately
  generated against mismatched context) scored independently by both a reviewer model and the real
  production judge (`gemini-3.5-flash`). **The discrimination check passes cleanly: all 5
  planted-unfaithful items were scored groundedness=1 by both scorers**, and `rejection_recall=1.0`
  in both directions — everything either scorer flagged as bad (≤2), the other flagged too. Where
  they disagreed (5 of 20 groundedness scores, kappa 0.30; only 1 of 20 answer_correctness scores,
  kappa 0.69), it was one-directional: the production judge was consistently *stricter*, correctly
  penalizing cases where the generator padded a correct core answer with fabricated elaboration
  (invented equations, thresholds, benchmark rows) the reviewer let pass — the judge being more
  reliable than the human-proxy, not less. See the 2026-08-20 (2) session log entry.
- [x] Run against current naive pipeline → record baseline in the Eval Results table below —
  **naive_dense_baseline: MRR 0.347 / Recall@5 0.489 / nDCG@10 0.418, n=89.** This is the real
  number; every earlier figure (0.508, 0.406, 0.384, 0.391, 0.343...) was a pilot or a draft-golden-set
  sanity check, not this.

### Phase 2 — Hybrid retrieval
- [x] Add BM25 sparse index alongside Chroma dense index — `server/core/bm25.py`, hand-rolled
  pure stdlib, **verified exactly against `rank_bm25`** (20/20 queries, worst |delta| 1.4e-14,
  reference lib installed temporarily and removed; `scripts/verify_bm25_against_rank_bm25.py`).
  Index is built from the Chroma collection itself, so both arms provably share one text and
  one id space.
- [x] Implement Reciprocal Rank Fusion (RRF) to combine dense + sparse results —
  `server/core/hybrid_retriever.py`, unweighted, `k=60` (Cormack et al. 2009) fixed a priori.
- [x] Re-run eval harness → record hybrid numbers — **`hybrid_rrf60`: MRR 0.485 / Recall@5 0.635
  / nDCG@10 0.571 (n=89)**, paired ΔnDCG@10 **+0.1532, 95% CI [+0.0971, +0.2099]**. The
  pre-registered rule fired; `DEFAULT_RETRIEVAL_MODE` is now `hybrid`.
- [x] Sensitivity check that the conclusion does not depend on the RRF constant —
  `scripts/rrf_sensitivity_sweep.py`, reported as a labelled diagnostic only.

### Phase 3 — Cross-encoder reranking
- [ ] Retrieve wide (top-50) via hybrid search
- [ ] Add BGE-reranker-v2-m3 (self-hosted) to rerank down to top-k (e.g. top-8)
- [ ] Re-run eval harness → record hybrid+rerank numbers, isolate the lift from reranking specifically

### Phase 4 — GraphRAG
- [ ] Stand up Neo4j (Docker)
- [ ] Extract entities/relations from corpus via LangChain's `LLMGraphTransformer`
- [ ] Build query router: simple lookup → vector path, multi-hop/relational → graph path (text-to-Cypher)
- [ ] Build 10-15 multi-hop eval questions that vector-only fails and graph succeeds on — record as a separate mini eval table

### Phase 5 — Multimodal ingestion
- [ ] Identify corpus docs with tables/scanned pages/charts
- [ ] Use Gemini (Flash for bulk, Pro for complex layouts) to parse these into structured text before chunking
- [ ] Re-run eval on the subset of questions that depend on this content

### Phase 6 — LangGraph orchestration
- [ ] Convert linear retrieve→rerank→generate into an explicit `StateGraph`
- [ ] Nodes: query classification/routing, retrieval (vector or graph), rerank, generate, groundedness self-check
- [ ] Add LangSmith tracing (or extend existing logging to structured tracing)

### Phase 7 — Serving, observability, polish
- [ ] Extend FastAPI routes for the new pipeline stages, add health checks
- [ ] Add Prometheus metrics: retrieval latency, rerank latency, generation latency, token cost per query
- [ ] Docker Compose for full stack (backend, Chroma/BM25 index, Neo4j, frontend)
- [ ] Final README: architecture diagram + eval results table + setup instructions
- [ ] Keep MIT license attribution to original author (Zlash65) in README credits

### Phase 8 — Interview packaging
- [ ] Finalize the before/after eval results table as the centerpiece artifact
- [ ] One-page architecture diagram
- [ ] Write out anticipated interview Q&A (see CLAUDE.md non-negotiables for the list)

---

## Corpus source & scope

**Sources:**
- Paper metadata + PDFs: **arXiv API** (`export.arxiv.org/api/query`) — categories `cs.CL`, `cs.IR`,
  relevant `cs.LG`, filtered by RAG/retrieval-adjacent keywords (hybrid retrieval, reranking,
  GraphRAG, long-context retrieval), last ~3 years.
- Citation edges: **Semantic Scholar Graph API** (arXiv-ID-keyed), not the arXiv API itself — arXiv's
  metadata has no reference list. This is what Phase 4's structured `cites` relation is built from.

**Pipeline (all scripted, `scripts/corpus_intake/`):**
1. `fetch_candidate_papers.py` — pulls ~80 recent candidates from arXiv, adds 14 hand-picked anchor
   papers, batch-queries Semantic Scholar for each paper's own reference list, and computes which
   references land on another paper already in the set (an internal citation edge).
2. `curate_candidates.py` — applies an explicit, reviewable keep-list on top of that: edge count is a
   signal, not the only one, since a paper's true topical fit only shows up in the abstract, not the
   title or the edge count alone (see the two entries below for what this caught in practice).
3. `freeze_corpus.py` — downloads the kept papers' PDFs, hashes each with sha256, and writes
   `data/papers/corpus_manifest.json` as the permanent provenance record (source URL, hash, download
   timestamp per paper) — the same manual-collection-vs-scriptable-sourcing question the earlier
   (discarded) corpus plan couldn't answer cleanly, resolved here by the domain choice itself.

**Anchor papers:** 14 total — 8 originally seeded for graph connectivity (DPR, RAG, REALM, ColBERT,
Self-RAG, GraphRAG, Lost in the Middle, Passage Re-ranking with BERT) plus 6 added after a first
external review surfaced them as directly relevant to specific phases (RAGAS → Phase 1 eval,
ColBERTv2/M3-Embedding → Phase 2/3, ColPali → Phase 5 multimodal, Adaptive-RAG/CRAG → Phase 6
routing). A 7th proposed addition was checked against the live arXiv API and turned out to be a
different, unrelated paper entirely — rejected rather than trusted on the strength of the claim
alone. In the frozen corpus, the anchors are the best-connected nodes by a wide margin (the RAG
paper alone has 26 internal citation edges), which turned out to make the originally-planned
"bridge papers to fix graph connectivity" unnecessary — the anchors already do that job once
citation lookups actually ran (see Key Architecture Decisions for how that got discovered).

**GraphRAG signal:** citation edges (`cites`) come from Semantic Scholar's structured data — no LLM
extraction needed, and also not interesting on their own (a citation lookup is answerable by any
citation API). The relations that actually justify running `LLMGraphTransformer` are the ones that
only exist in unstructured paper text: `evaluates_on` (same benchmark), `extends`/`outperforms`
(method comparisons). That split — structured backbone + LLM-extracted semantic layer — is the
answer to "why GraphRAG instead of just querying a citation API."

**Multimodal signal (Phase 5):** benchmark/result tables and architecture-diagram figures, all
clean/born-digital, not scanned — a tidier version of the multimodal challenge, worth being
upfront about in the interview narrative rather than implying it's the same difficulty as messy
real-world scans.

**Final frozen composition:** 45 papers (14 anchor + 31 recency-window), 112 internal citation
edges, 89.0 MB of PDFs, 43/45 found in Semantic Scholar's index (the 2 misses are papers published
within the last day or two — too fresh for S2 to have indexed yet, not a relevance problem).
Provenance for every paper (arXiv ID, title, authors, source URL, sha256, download timestamp,
citation count) is in `data/papers/corpus_manifest.json`.

**Ingestion (into the vector store, per provider — reference number for later phases):** 45 PDFs →
736 page-documents (`PyPDFLoader`) → **2131 chunks** (`TokenTextSplitter`, chunk_size=500,
overlap=50). This 2131 figure is the same regardless of provider, since chunking happens before
the embedding step splits by provider. As of 2026-08-05: groq (local
`sentence-transformers/all-MiniLM-L12-v2`) collection populated, 2131/2131 chunks, at
`server/data/groq_vector_store`. gemini (`gemini-embedding-001` API) not yet populated — see Open
questions/TODOs.

---

## Extension points (file → phase map)

Confirmed the live repo tree matches CLAUDE.md's file map exactly (no drift yet — 10 `.py`
files, nothing extra). Mapping below is which *existing* file each phase edits, plus which
*new* file each phase introduces.

**Files that get edited in place, in order:**

| File | Current role (Phase 0 baseline) | First touched in | What changes |
|---|---|---|---|
| `server/core/vector_database.py` | Chroma-only ops: embed, upsert, `similarity_search` | Phase 2 | Becomes `HybridRetriever` — adds a BM25 index alongside Chroma, fuses via RRF. `find_similar_chunks` stops being a thin Chroma passthrough. |
| `server/core/llm_chain_factory.py` | Builds one retriever → stuff-chain → LLM, linear | ~~Phase 3~~ **Phase 2 (actual)** | **Moved a phase earlier.** Wiring hybrid into the live app means `/chat` needs a retriever that is not `vectorstore.as_retriever(...)`, so `build_llm_chain` gained a `retrieval_mode` parameter selecting the hybrid LangChain adapter. Phase 3 still inserts the rerank step here as planned. |
| `server/core/llm_chain_factory.py` (again) | — | Phase 6 | Wrapped/replaced — chain construction moves from a function returning a LangChain `Runnable` to nodes wired into a LangGraph `StateGraph`. |
| `server/core/vector_database.py` (again) | — | Phase 4 | Gains a graph query path — `find_similar_chunks` no longer the only lookup; a router decides vector vs. graph. |
| `server/core/document_processor.py` | `PyPDFLoader` flattens every page to plain text, then `TokenTextSplitter` | Phase 5 | Gains a pre-chunking step: Gemini-based parsing for benchmark/result tables and architecture-diagram figures before the existing `split_documents_to_chunks` runs. |
| `server/api/routes.py` | 6 routes: health, llm options/models, upload, count, search, chat | Phase 7 | Extended for new pipeline stages (rerank toggle, graph vs vector routing, health checks per new dependency). |
| `server/config/settings.py` | Static `MODEL_OPTIONS`/`VECTORSTORE_DIRECTORY` dicts | Phase 4, 7 | Gains Neo4j connection config (Phase 4), Prometheus/observability config (Phase 7). |
| `client/` (all) | Streamlit UI, thin wrapper over the API | Phase 7 | Left mostly as-is until Phase 7 polish, per CLAUDE.md. |

**New files/modules introduced per phase (all under `server/core/` unless noted):**

| Phase | New module | Purpose |
|---|---|---|
| 1 | `eval/` (golden question set + scorer) | MRR / Recall@k / nDCG@k against whatever retriever exists at the time — this is what makes every later phase's "did it help" question answerable instead of vibes-based. |
| 2 | `hybrid_retriever.py` + `bm25.py` | BM25 + dense + RRF fusion logic (the mechanics `vector_database.py` calls into). **Split into two files, deviating slightly from the planned single module:** `bm25.py` is the tokenizer + scorer and imports nothing outside the stdlib, so its 30 tests run with no vectorstore — the same purity constraint `eval/metrics.py` is held to, and for the same reason. `hybrid_retriever.py` owns everything that touches Chroma, plus the RRF fusion and the LangChain adapter. |
| 2 | `eval/compare_runs.py` | Paired-bootstrap comparison of two run artifacts, stratified, plus the per-arm gold attribution and the mechanical application of the pre-registered decision rule. A single run prints a number; the phase's actual claim is the *difference* between two runs and whether it survives n=89. |
| 3 | `reranker.py` | BGE-reranker-v2-m3 wrapper — takes hybrid_retriever's wide top-50, returns top-k. |
| 4 | `graph_store.py` | Neo4j connection; structured `cites` edges ingested from Semantic Scholar Graph API (no LLM needed); `LLMGraphTransformer` ingestion for text-derived relations; Cypher QA chain. Schema: paper/author/method/dataset/benchmark entities, `cites` (structured) + `evaluates_on`/`extends`/`outperforms` (LLM-extracted) relations. |
| 5 | `multimodal_ingest.py` | Gemini (Flash/Pro) parsing of benchmark/result tables and architecture-diagram figures, called from `document_processor.py` before chunking. |
| 6 | `graph_pipeline.py` (or similar) | The explicit LangGraph `StateGraph` — nodes: classify/route, retrieve (vector or graph), rerank, generate, groundedness self-check. |
| 7 | Docker Compose additions | Neo4j service, and a new vector DB service if Chroma is outgrown. |

**Reading order this implies:** Phases 1→3 are almost entirely inside `vector_database.py` +
`llm_chain_factory.py` (the retrieval/generation core) and are additive — nothing about the
API surface or document ingestion needs to change yet. Phase 4 is the first one that also
touches `document_processor.py`'s *sibling* concern (entity extraction happens off the same
chunks, not instead of them) and `routes.py` (routing logic). Phase 5 is the only phase that
changes *pre-chunking* behavior. Phase 6 is a structural rewrite of chain composition, not a
new capability — it's `llm_chain_factory.py`'s internals reorganized around what by then are
several already-built pipeline stages.

---

## Key architecture decisions

| Decision | Options considered | Chosen | Why |
|---|---|---|---|
| Dependency management | Two `requirements.txt` files (server + client) vs single root `pyproject.toml` | `pyproject.toml` (setuptools backend) | Cleaner structure and locking; started with the requirements.txt split but consolidated once both client and server deps needed to be managed together |
| Citation data source | (a) arXiv API alone, (b) Semantic Scholar Graph API, (c) OpenAlex | Semantic Scholar Graph API | arXiv's own metadata API has no reference list — no citation edges to build a graph from without a second source. Semantic Scholar indexes arXiv IDs directly and is free/scriptable, same public-data profile as arXiv itself |
| Graph relation strategy | (a) Citation edges (`cites`) only, (b) citations + LLM-extracted semantic relations (`evaluates_on`, `extends`, `outperforms`) | (b) | A pure citation graph is answerable by a citation API directly — it doesn't need GraphRAG or justify running `LLMGraphTransformer`. The LLM-extracted layer (relations that only exist in unstructured paper text) is what actually demonstrates GraphRAG's value over a metadata lookup, and is the direct answer to "why not just query a citation API" |
| Corpus recency window vs. graph connectivity | (a) Last ~3 years only, matching the RAG/retrieval subtopic scope, (b) same, plus a small seeded set of older foundational papers (DPR, RAG, cross-encoder rerankers, GraphRAG) | (b) | A recency-only corpus risks a citation graph where most references point to foundational work outside the corpus (BM25, DPR, the original RAG paper all predate a 3-year window), leaving few internal edges to traverse. Seeding anchor papers is cheap insurance against Phase 4 having a sparse or disconnected graph |
| Semantic Scholar query shape | (a) One GET per paper (`/paper/arXiv:{id}`), fetching both `references` and `citations`, (b) one GET per paper, `references` only, (c) batched POST (`/paper/batch`) for the whole set, `references` only | (c) | (a) crashed outright: a highly-cited anchor's `citations` list (who cites it) is unbounded and made a single request time out. (b) fixed the timeout but still made ~88 sequential unauthenticated requests, which got rate-limited into near-total failure (most requests 429'd even with retry/backoff). (c) does the same lookups in 2 requests instead of 88 and finished in under 3 minutes. The underlying insight both fixes rely on: "A cites B" is fully recoverable from A's own (small, bounded) reference list — the reverse `citations` direction is never actually needed |
| Whether "bridge papers" were needed for graph connectivity | An earlier (external) review proposed adding papers specifically to connect otherwise-isolated anchor nodes, based on an internal-edge count of zero across the anchor set | Not needed — rejected once real data was available | That "zero edges" figure came from a smoke-test run where citation lookups had been skipped entirely (`--skip-s2`), not from citation data showing real disconnection. Once the batch lookup actually ran, the anchors turned out to already be densely connected to each other and to the recent pool (the RAG paper alone has 26 internal edges) — the connectivity problem the bridge papers were meant to solve didn't exist once the tooling was fixed. Lesson: a "zero" result from a pipeline is a signal to check whether the pipeline ran, not to accept the zero at face value |
| Corpus curation signal | (a) Internal citation edge count alone (drop zero-edge papers), (b) edge count plus a manual title read, (c) edge count plus reading each paper's actual abstract | (c) | Edge count alone is misleading in both directions: several genuinely on-topic papers (e.g. very recently published ones) showed 0 edges purely because Semantic Scholar hadn't indexed them yet, not because they're irrelevant; and a title read alone let one paper through that turned out, by its abstract, to be about a different problem entirely (a recommender-systems explainability paper, not document retrieval for QA, despite a title containing "RAG Framework" and "Knowledge Graph"). Reading the actual abstract for every non-anchor candidate caught that before the freeze, not after |
| Chunk identity | (a) Keep Chroma's default random UUID4s, (b) derive a content hash (sha256 of chunk text) at eval time, (c) deterministic positional ids `{arxiv_id}::p{page}::c{ordinal}`, written at ingest | (c) | (a) was not viable at all: `langchain_community` 0.4.2 rebuilds Documents from `page_content` + `metadata` and never reads the ids column, so retrieved chunks carried **no identity whatsoever** — the golden set's `expected_chunk_ids` field had no referent. Between (b) and (c): both are stable, but they fail differently. Phase 5 (multimodal ingestion) will re-chunk the corpus, and when that happens a positional id still says "page 2 of DPR" so a stale gold id can be re-resolved by hand at page granularity; a stale content hash is unrecoverable. Ids are written BOTH as Chroma's `ids=` (making re-ingest an idempotent upsert instead of a duplicate-append) and into `metadata["chunk_id"]` (the only copy that survives retrieval) — neither alone suffices. Phase 2's RRF and Phase 3's reranking both join *ranked lists of documents*, so this cost was going to be paid regardless; paying it before authoring 75-100 questions avoids redoing them |
| Relevance model for the golden set | (a) One gold chunk per question, (b) a flat set of gold chunks, (c) per-question `any_of`/`all_required` flag, (d) a list of disjoint **equivalence groups**, credit once per group | (d) | (a) is dishonest given `chunk_overlap=50`: an overlapping neighbour that fully answers the question scores zero. (b) rewards retrieving redundant near-duplicates (+0.081 nDCG in a measured example) and cannot express multi-hop. (c) was the initial design and handles both common cases, but cannot express the general one — a cross-paper multi-hop question where *each hop* also has adjacent alternatives. (d) subsumes all three: `any_of` = one group, `all_required` = N singletons. It is also not bespoke — group recall is **S-recall/subtopic recall** (Zhai et al. 2003) and the gain scheme is **α-nDCG at α=1** (Clarke et al. 2008). Pinning α=1 over *disjoint* groups gives IDCG a closed form and makes nDCG ≤ 1 provable, where general α-nDCG needs a greedy approximation of an NP-hard ideal and can exceed 1. `semantics` is kept on disk as a stratification label but metrics never branch on it, so there is exactly one scoring path |
| MRR definition for multi-hop questions | (a) Reciprocal rank of the FIRST covered group, (b) mean of per-group reciprocal ranks, (c) `RR_cov` = reciprocal of the rank at which the LAST group is covered | (c) | On a 2-hop question that finds hop A at rank 1 and never finds hop B — a question the system fundamentally cannot answer — (a) scores **1.0** and (b) scores **0.5**. Both hand out credit for zero answerability, and they do it specifically on the multi-hop questions that exist to expose the baseline's weakest point, which is precisely where a flattering number does the most damage to Phase 4's story. (c) scores 0.0, and is also the operationally meaningful quantity: since the chain stuffs top-k into one prompt, the question is answerable iff all evidence is within k. (c) additionally reduces to *textbook* MRR on single-group questions, so the logged figure stays comparable to published MRR. (a) and (b) are still computed and logged as diagnostics — (b) retains a gradient useful for debugging — but never as the headline |
| Aggregation across questions | (a) Micro-average (pool groups across questions), (b) macro-average with a single pooled headline, (c) macro-average, always reported stratified by `any_of` / `all_required` | (c) | Micro weights a 3-hop question 3× a single-hop one, letting ~10 multi-hop questions swing a 90-question headline. But pooling alone hides the signal that matters: modelled on ~10 multi-hop questions going from unanswerable to answerable at depth 3, the `all_required` stratum moves 0.000 → 0.133 (a step change from total failure) while the pooled headline moves +0.015 — noise at n≈90. Reporting only the pooled number would mean building the Phase 4 graph path, making it work, and reporting a delta a skeptical interviewer would correctly dismiss |
| Golden-set gold labelling | (a) Trust the source chunk as gold by construction, (b) also verify the source chunk answers its own question, (c) (b) plus pooling retriever-surfaced candidates as additional gold | (b) | The pilot caught (a) failing outright: the generator invented questions from passages that merely *mention* a topic. One gold chunk was a related-work citation list ("corrective RAG14, Self-RAG23…") for a question about how negatives are chosen; another was a paper's **title page** for a question about how its system works — while retrieval returned that paper's Conclusion, which answers it exactly, and was scored a total miss. The bug was an asymmetry: adjacent candidates faced a strict admission test while the source was assumed correct. Both now face it. (c) was rejected on principle: using the dense retriever under test to build its own relevance judgments biases the eval toward that retriever, and a chunk naive dense never surfaces could never become gold — so a Phase 2 hybrid system that finds it would get zero credit, working directly against the result this project exists to demonstrate. Re-pooling is deferred to Phase 2/3 under an explicit protocol (see Open questions/TODOs) |
| Question specificity filter | (a) No filter, (b) reject near-duplicate questions only, (c) require a named method/system/dataset AND reject near-duplicates — applied to bulk generation only | (c), scoped | Three of 18 pilot questions were variants of "how are negative examples chosen during training", and two questions retrieved the *same* top-1 chunk, so at most one could ever score — a guaranteed miss that says nothing about retrieval. Requiring a named entity makes each question identify a single passage. Deliberately scoped to the ~75 LLM-generated questions and NOT the ~20 hand-written hard ones: a good paraphrase question intentionally avoids naming the method by its exact term (asking how a system decides which examples are wrong, rather than "how are negative examples chosen"), and forcing an entity into it would defeat the reason that question exists. Known limitation to state plainly: this skews the bulk set toward entity-lookup questions, which dense retrieval handles relatively well |
| Groundedness judge | (a) RAGAS library, (b) manual rubric on a sample, (c) hand-rolled LLM-as-judge, generator ≠ judge family | (c) | (a) pins its own LangChain version range — a demonstrated, not hypothetical, risk in this repo, which already spent a session repointing every `from langchain.x` import after the 1.0 split — and "somewhat black-box" is a weak answer to "how is faithfulness actually computed". (b) cannot be re-run after every phase, which is exactly the CLAUDE.md re-validation non-negotiable. (c) generates with a Groq model and judges with Gemini `gemini-3.5-flash`: different model family (the strongest available defence against self-preference bias) and judge at least as capable as the generator (a weaker judge grading a stronger generator is unreliable). `groundedness` is scored against the contexts actually passed to the LLM, kept separate from `answer_correctness` vs the reference — conflating them would hide the most interesting RAG failure mode, an answer perfectly faithful to wrongly-retrieved context |
| Golden-set second-pass review | (a) Ship after one self-audit (full-text re-read of every miss/sample/multi-chunk group), (b) also run an independent second model over the same content and treat its output as authoritative, (c) (b) but re-verify every flag against source text before applying any fix | (c) | (a) alone had already let two borderline cases through once (`b0058`/`b0059`-era thin and mis-sourced gold) — a second pass from the *same* reviewer (me) is correlated with the same blind spots. (b) was rejected after the reviewer model's own summary arithmetic didn't reconcile with its row-level verdicts on the very first check (it reported totals that didn't match a recount of its own table) — trusting an LLM's self-tally is the same "0 internal citation edges" failure mode in a new costume. (c): of 12 flagged bulk items, 10 were confirmed real on independent re-read and fixed (5 repointed to a chunk that actually explains the mechanism, 2 dropped as unfixable in-corpus, 3 were question/answer wording fixes), 2 were rejected as over-strict (chunks already supported the question at the same depth as several accepted items). Net effect: 75 → 71 → 69 bulk questions, each drop/repoint independently justified against source text, not against the reviewer's say-so |
| Judge-validation reviewer model | (a) The user hand-scores the validation batch personally, (b) a second AI model scores it, standing in for the human step | (b), with a stated caveat | The project's own design explicitly wants judge validation to catch blind spots the production judge (`gemini-3.5-flash`) might have — using a *different* Gemini model as the validator is weaker than an independent human, since family-correlated blind spots wouldn't be caught. Proceeded anyway (the same tradeoff already made for golden-set review, where it worked), and the completed comparison (2026-08-20 (2)) backs that up: all 5 deliberately-unfaithful items were scored groundedness=1 by *both* the reviewer and the real judge independently, `rejection_recall=1.0` in both directions on both metrics, and every one of the 5 groundedness disagreements ran in the same direction — the production judge stricter, correctly penalizing fabricated elaboration the reviewer let pass, each one individually verified against the actual generated text rather than trusted on the judge's say-so. The family-correlation risk this row worried about didn't materialize here; if anything the reviewer under-flagged relative to the real judge, not over-flagged |
| BM25 implementation | (a) `rank_bm25` (`BM25Okapi`), (b) LangChain's `BM25Retriever`, (c) hand-rolled, pure stdlib | (c) | There is no performance argument for a library at this scale — measured, not assumed: 2131 chunks / 459K tokens / 37.5K vocabulary indexes in ~0.7s including the Chroma read. So a dependency buys only opacity, in a project whose eval scorer is deliberately pure stdlib for the same reason and which already rejected RAGAS partly for being "somewhat black-box". (b) additionally wraps (a) and couples the sparse arm to `langchain-community`, which now prints a sunset-deprecation warning on import. The obvious risk of hand-rolling — a subtle IDF or length-normalization bug that still returns plausible documents while quietly *understating* the hybrid lift — is closed by `scripts/verify_bm25_against_rank_bm25.py`: `rank_bm25` installed temporarily (never a dependency), fed the same tokenizer output, exact score agreement asserted over the full corpus on 20 real golden-set queries. Result: 20/20 exact, worst |delta| 1.4e-14 (float noise). The claim is "hand-rolled **and** verified against the reference implementation", not "hand-rolled" |
| BM25 tokenization scheme | (a) naive lowercase + split on non-alphanumerics, (b) same plus stemming and a stopword list, (c) NFKD-normalize → casefold → strip combining marks → tokenize `[a-z0-9]+([._-/][a-z0-9]+)*` → emit compounds **and** their sub-tokens; no stemming, no stopwords | (c) | BM25's term matching is only as good as the split, and this is a first-class decision that simply does not arise with dense retrieval (the encoder absorbs it). Two corpus-measured hazards drove it: **ligatures** — 212 of 2131 chunks (~10%) contain PDF ligature glyphs, so an ASCII tokenizer shreds "identification" into `identi` + `cation` across a tenth of the corpus (NFKD also folds accented author names); and **hyphenated compounds** — `retrieval-augmented` ×380, `multi-hop` ×261, `end-to-end` ×152, `top-k` ×94, `re-ranking` ×92, plus method names like `gfm-rag`. Indexing the compound *and* its parts lets "retrieval augmented generation" match "retrieval-augmented generation" while the intact compound stays a high-IDF term rewarding the exact form (visible live: `gfm-rag`=7.48 alongside `gfm`=7.32 in the same hit). Stemming and stopword removal were rejected on the same principle: BM25 is in the pipeline *because* it catches exact terms a bi-encoder blurs past (ColBERTv2→ColBERT, nDCG@10→"a ranking metric"), and an aggressive tokenizer spends exactly the advantage it was added for — morphological variation is the dense arm's job in the fusion. IDF already drives near-universal terms to ~0, making a stopword list a redundant knob that can only break exact phrases. Pinned as behaviour by `eval/tests/test_bm25.py`, versioned by `TOKENIZATION_VERSION`; plural folding is a candidate future *measured* change, not a silent one |
| Sparse index source of truth | (a) re-parse the PDFs and re-chunk for BM25, (b) build the BM25 index from the Chroma collection's own stored documents | (b) | RRF joins two ranked lists of *the same* documents, so the arms must agree on both the text and the ids. (a) reproduces the chunking pipeline a second time, which means any future re-chunk (Phase 5 will re-chunk) can silently desynchronise the two indexes — and the failure is invisible, since both arms keep returning plausible results and the fusion keeps returning a ranking. (b) makes divergence structurally impossible and inherits the Phase 1 deterministic chunk ids for free. Cost is one collection read (0.51s) plus tokenization (0.12s) per process, cached per provider — and invalidated on upload, because a stale sparse arm fused with a fresh dense arm is retrieval over two different corpora |
| RRF constant and weighting | (a) sweep `k` and dense:sparse weights on the golden set, keep the best, (b) fix `k=60` from Cormack et al. 2009 with no sensitivity analysis, (c) fix `k=60` a priori, run the sweep separately as a labelled diagnostic that never feeds the reported pipeline | (c) | (a) is textbook test-set leakage: tuning a hyperparameter on the only 89 questions the headline result is claimed from, with no held-out set to show it generalises. It yields a bigger delta and a materially weaker claim at the same time. (b) leaves "is the result fragile to that constant?" unanswered. (c) reports the standard, citable, unweighted formula as the shipped system and separately demonstrates the conclusion survives `k ∈ {10,30,60,100}`. Weighted-RRF variants are kept in a table explicitly headed *exploratory* so there is never ambiguity about which version produced the recorded number. "I used the constant from the original paper and separately confirmed the result isn't fragile to it" beats a larger number that cannot be defended |
| Whether hybrid becomes the live default | (a) flip on faith once implemented, (b) decide after seeing the eval numbers, (c) pre-register the decision rule in this log *before* running the eval, then apply it mechanically | (c) | (b) is the same failure as (a) in the RRF row above, one level up: a threshold chosen after seeing the result is not a threshold. The rule (pooled ΔnDCG@10 > 0, paired-bootstrap 95% CI lower bound > 0, no stratum regressing) was written into the Eval Results section before the hybrid run executed, names a single primary metric so a null result cannot be rescued by whichever secondary metric happened to move, and is applied by `eval/compare_runs.py`, which exits non-zero when it does not fire. The paired bootstrap is the point: at n=89 a raw "+0.04" is not evidence, and `paired_delta_ci` has been sitting unused in `metrics.py` since Phase 1 waiting for exactly this comparison |

*(Add a row every time you make a real architectural choice — vector DB, embedding model, reranker, etc. This table is your interview cheat sheet.)*

---

## Eval results

**Baseline recorded 2026-08-20** against `golden_set.v1.json` (89 questions, frozen that day — see
Phase 1 status board and the session log entry). This is the first valid number: every earlier figure
(the 2026-08-14 pilot's 0.508, and every draft-golden-set sanity run during the audit) was scored
against a golden set later found to have label defects or was explicitly a draft, never entered here.

**Metric definitions** (see `eval/metrics.py`): MRR is `RR_cov`, the reciprocal of the rank at which
the *last* required gold group is covered — on single-gold questions this is textbook MRR. Recall@k
is group (subtopic) recall; on single-gold questions it is hit-rate@k. Both are macro-averaged over
questions. `Gold set` records which version of the golden set the row was scored against.

| Pipeline variant | Gold set | MRR | Recall@5 | nDCG@10 | Notes |
|---|---|---|---|---|---|
| Naive dense-only (baseline) | v1 | 0.347 | 0.489 | 0.418 | n=89, groq/MiniLM-L12-v2, depth=50, variant `naive_dense_baseline` |
| + Hybrid (dense + BM25 + RRF) | v1 | **0.485** | **0.635** | **0.571** | n=89, unweighted RRF k=60, pool=50/arm, variant `hybrid_rrf60`. ΔnDCG@10 **+0.153, 95% CI [+0.097, +0.210]** paired |
| + Cross-encoder rerank | | | | | |
| + GraphRAG routing (multi-hop subset) | | | | | |
| + Multimodal ingestion (affected subset) | | | | | |

**Phase 2 pre-registered decision rule (written 2026-08-20, BEFORE the hybrid run).**
Recorded here first so it cannot be reinterpreted once the number is visible — the same
discipline as refusing to tune the RRF constant on this golden set.

- **Primary metric:** pooled (`all` stratum) **nDCG@10**, paired bootstrap over the same 89
  questions (`eval/metrics.py::paired_delta_ci`, 10 000 iters, seed 0, 95%).
- **Hybrid becomes the app default (`DEFAULT_RETRIEVAL_MODE`) iff** mean paired ΔnDCG@10 > 0
  **and** the 95% CI lower bound > 0 **and** neither stratum shows a clear regression
  (per-stratum ΔnDCG@10 CI upper bound ≥ 0).
- **If the CI includes 0:** record "no measurable difference at n=89", keep `dense` as the
  default, keep the hybrid path available behind the `retrieval_mode` flag, and report it
  either way. A null result measured properly is still a result.
- MRR, Recall@5 and Sufficiency@5 are reported but do **not** vote. Choosing the winner from
  whichever metric happened to move is how a null result becomes a positive one.
- `eval/compare_runs.py` applies this rule mechanically and exits non-zero when it does not fire.

**Phase 2 numbers are a lower bound, not a final figure.** The re-pooling protocol below does
not execute until Phase 3 exists, so the golden set currently only labels chunks that
*dense-only* retrieval surfaced. A genuinely correct chunk that only BM25 reaches scores zero
today. Whatever lift Phase 2 measures is therefore a floor, and a larger delta after the v2
re-pool is not a contradiction of the row above.

**Comparability rule (decided 2026-08-14).** Rows are only comparable within the same `Gold set`
version. When the golden set changes — including the Phase 2/3 re-pooling below — every prior variant
must be **re-scored against the new version** and its row updated, or moved to a clearly-labelled
archive table. Without this, "nDCG improved by X%" silently conflates a real retrieval improvement
with a change in what counts as correct.

**By-stratum table** (populated alongside the pooled rows above; `all_required` is where Phase 4 must
show its lift, and it is diluted ~9× in the pooled column):

| Pipeline variant | Gold set | Stratum | n | MRR | Recall@5 | Sufficiency@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| Naive dense-only (baseline) | v1 | any_of | 79 | 0.385 | 0.519 | 0.519 | 0.436 |
| Naive dense-only (baseline) | v1 | all_required | 10 | 0.044 | 0.250 | 0.100 | 0.272 |
| + Hybrid (RRF k=60) | v1 | any_of | 79 | 0.534 | 0.684 | 0.684 | 0.598 |
| + Hybrid (RRF k=60) | v1 | all_required | 10 | 0.095 | 0.250 | **0.100** | 0.357 |

**Phase 2 paired comparison** (`eval/compare_runs.py`, 10 000 bootstrap iters, seed 0, same 89
questions). Every pooled metric moves well outside its interval, and no stratum regresses:

| Stratum | ΔMRR | ΔRecall@5 | ΔSufficiency@5 | ΔnDCG@10 (primary) |
|---|---|---|---|---|
| all (n=89) | +0.138 [+0.082, +0.195] | +0.146 [+0.062, +0.236] | +0.146 [+0.067, +0.236] | **+0.153 [+0.097, +0.210]** |
| any_of (n=79) | +0.149 [+0.085, +0.212] | +0.165 [+0.076, +0.266] | +0.165 [+0.076, +0.266] | +0.162 [+0.103, +0.221] |
| all_required (n=10) | +0.051 [+0.002, +0.115] | +0.000 [−0.150, +0.150] | +0.000 [+0.000, +0.000] | +0.084 [−0.078, +0.245] |

Per-question movement: **58 up, 5 down, 26 unchanged; 23 total misses rescued, 0 previously-found
questions newly missed.** The last figure matters — a fusion that traded one set of hits for another
would show a similar mean delta with a very different risk profile.

**Which arm actually reached the evidence** (106 gold chunks reached across the set):

| Reached by | Gold chunks | Share |
|---|---|---|
| both arms | 72 | 67.9% |
| dense only | 5 | 4.7% |
| **BM25 only** | **29** | **27.4%** |

**28 of 89 questions had gold evidence that only the BM25 arm reached at any depth.** That is the
substantive Phase 2 finding, and it is stronger than the aggregate delta: this evidence was not
merely ranked poorly by dense retrieval, it was *absent from its top-50 entirely* — so no amount of
Phase 3 reranking could have recovered it, since a reranker only reorders what retrieval already
found. It also confirms the two arms are complementary rather than redundant (only 4.7% of gold was
dense-exclusive).

**Note on `all_required`:** MRR moves (+0.051) but **Sufficiency@5 does not move at all** — still
0.100, with a CI of exactly [0, 0]. Hybrid retrieval gets *closer* on multi-hop questions without
making them answerable: nine of ten still fail to land all required evidence within the depth a
stuffed-context chain uses. That headroom is untouched and remains Phase 4's to close. Reporting the
pooled +0.153 alone would have hidden this completely.

**RRF sensitivity — DIAGNOSTIC ONLY, not a tuning result** (`scripts/rrf_sensitivity_sweep.py`;
both arms retrieved once per question and re-fused per config, so these are exact). The shipped
pipeline is the unweighted `k=60` row, fixed before any of these numbers existed:

| Config | MRR | Recall@5 | nDCG@10 |
|---|---|---|---|
| unweighted k=10 | 0.520 | 0.702 | 0.613 |
| unweighted k=30 | 0.487 | 0.635 | 0.577 |
| **unweighted k=60 — SHIPPED** | **0.485** | **0.635** | **0.571** |
| unweighted k=100 | 0.484 | 0.624 | 0.571 |
| k=60, weighted 2:1 toward dense *(exploratory, not the shipped formula)* | 0.449 | 0.573 | 0.526 |
| k=60, weighted 1:2 toward BM25 *(exploratory, not the shipped formula)* | 0.511 | 0.652 | 0.588 |

Pooled nDCG@10 spread across `k ∈ {10,30,60,100}` is **0.042**, against a dense→hybrid delta of
**0.153** — the Phase 2 conclusion is not an artifact of the fusion constant. Note that the shipped
`k=60` is **not** the best-scoring row: `k=10` scores +0.042 higher. That number is deliberately not
claimed. Selecting it after the fact would be fitting a hyperparameter to the same 89 questions the
result is reported on, with nothing held out to show it generalises — the ~0.04 left on the table is
the price of a defensible claim, and is itself the answer to "how do you know you didn't tune to
your eval set".

The gap is exactly the story this golden set exists to tell: on `all_required` (genuine multi-hop,
both/all pieces of evidence needed), `Sufficiency@5` is 0.100 — nine of ten multi-hop questions are
*unanswerable* by naive dense retrieval within the depth a stuffed-context chain actually uses. That
headroom is what Phase 4 (GraphRAG) has to move, and it would have been invisible in the pooled row
alone (0.347 MRR looks unremarkable; 0.044 does not).

---

## Session log

*(Append one entry per work session — date, what was done, files touched, what's next. Claude Code should update this at the end of every session.)*

### YYYY-MM-DD
- **Done:**
- **Files touched:**
- **Eval numbers (if changed):**
- **Next:**

### 2026-07-26
- **Done:** Consolidated dependency management into a single root `pyproject.toml` (setuptools build backend), deduplicating deps from `server/requirements.txt` and `client/requirements.txt` (no version pins existed, so no conflicts to resolve). Deleted both old requirements.txt files. Updated CLAUDE.md's file map and Commands section to reflect the new install path.
- **Files touched:** `pyproject.toml` (new), `server/requirements.txt` (deleted), `client/requirements.txt` (deleted), `CLAUDE.md`, `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — no retrieval-touching change
- **Next:** Continue with Phase 0 (baseline the existing codebase, decide and shortlist the corpus)

### 2026-07-26 (2)
- **Done:** Traced both base-repo flows end to end without changing code: (1) PDF upload —
  `routes.py` → `document_processor.py` (validate/save/load/chunk) → `vector_database.py`
  (embed per provider, Chroma insert/append), and (2) question answering — `routes.py` →
  reload Chroma → `llm_chain_factory.py` (top-3 dense retriever, stuff-documents chain, LLM
  call) → answer extraction. Wrote the architecture summary into the Phase 0 blank and checked
  off the three tracing/summary items on the status board.
- **Files touched:** `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — no retrieval-touching change, read-only exploration
- **Next:** Get backend/frontend running locally, decide and shortlist the corpus,
  note extension points per CLAUDE.md file map to close out Phase 0

### 2026-07-26 (3)
- **Done:** Full import-compatibility audit against the installed environment (LangChain 1.0
  generation: `langchain-core` 1.5.1, `langchain-community` 0.4.2, `langchain-classic` 1.0.8,
  `langchain-text-splitters` 1.1.2 — no bare `langchain` package installed at all). Root cause of
  the runtime error: every old-style `from langchain.xxx import yyy` in the base repo
  (`langchain.prompts`, `langchain.chains`, `langchain.embeddings`, `langchain.document_loaders`,
  `langchain.text_splitter`) fails with `ModuleNotFoundError: No module named 'langchain'` because
  those symbols now live in split packages (`langchain_core`, `langchain_classic`,
  `langchain_community`, `langchain_text_splitters`). Repointed all six imports at their modern
  homes, added `langchain-core`/`langchain-classic`/`langchain-text-splitters` as explicit
  `pyproject.toml` dependencies (previously only present transitively via `langchain-community`),
  and replaced the deprecated `@app.on_event("startup")` with a `lifespan` context manager in
  `server/main.py`. Verified with `py_compile` on every project `.py` file, then booted
  `uvicorn main:app` for real — startup lifespan ran (both provider vectorstores initialized) and
  `GET /health` returned 200.
- **Files touched:** `server/core/llm_chain_factory.py`, `server/core/vector_database.py`,
  `server/core/document_processor.py`, `server/main.py`, `pyproject.toml`
- **Eval numbers (if changed):** N/A — import/runtime compatibility fix only, no retrieval logic changed
- **Next:** `langchain_community.embeddings.HuggingFaceEmbeddings` and
  `langchain_community.vectorstores.Chroma` still work but are deprecated in favor of dedicated
  `langchain-huggingface` / `langchain-chroma` packages (not yet installed) — worth migrating
  before Phase 2 hybrid retrieval work touches `vector_database.py` anyway. Otherwise continue
  Phase 0: get frontend running, decide and shortlist the corpus.

### 2026-07-26 (4)
- **Done:** Fixed a live 404 hit while using the Streamlit UI with the Gemini provider:
  `models/embedding-001` (the base repo's original hardcoded Gemini embedding model, only
  used in application code at `server/core/vector_database.py:27`) has been retired by
  Google — `GoogleGenerativeAIEmbeddings` has no client-side model validation, so the
  invalid ID reached the API and 404'd on every PDF upload. Confirmed the currently valid
  replacement by calling the live `ListModels` endpoint with the project's own
  `GOOGLE_API_KEY`: three embedContent-capable models exist today —
  `gemini-embedding-001` (GA, text, 2048 input tokens — direct successor, same version
  suffix as the old model), `gemini-embedding-2` (GA, multimodal, 8192 tokens), and
  `gemini-embedding-2-preview`. Chose `gemini-embedding-001` as the minimal, low-risk swap
  since this project's corpus is currently text-only (multimodal handling is separately
  planned for Phase 5 via Gemini table parsing, not the embedding model). Verified with a
  live `embed_query` call (no 404), then exercised the real endpoint end-to-end: booted
  `uvicorn`, uploaded a real PDF via `POST /upload_and_process_pdfs` with
  `model_provider=gemini`, and confirmed `POST /vector_store/search` returned the correct
  chunk.
- **Files touched:** `server/core/vector_database.py`, `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — provider-specific embedding model fix, no retrieval
  logic or ranking changed
- **Next:** Continue Phase 0: get the Streamlit frontend running end-to-end, decide and
  shortlist the corpus.

### 2026-07-26 (5)
- **Done:** Fixed a second, distinct 404 hit immediately after the embedding fix above — this
  one from the *chat* model, not the embedding model: `gemini-2.5-flash` (one of two hardcoded
  entries in `server/config/settings.py`'s `MODEL_OPTIONS`) has been retired by Google for new
  users. Rather than trust Google's `ListModels` listing alone (learned from the embedding fix
  that a model can appear there and still fail on actual use), verified every candidate with a
  real `invoke()`/chat-completion call using the project's own `GOOGLE_API_KEY`/`GROQ_API_KEY`:
  `gemini-2.0-flash` turned out to be 429 RESOURCE_EXHAUSTED ("exceeded your current quota" —
  effectively dead weight on this key/tier despite being listed), `gemini-2.5-flash` 404'd as
  reported, and Groq's `llama3-70b-8192` (the other stale entry) is fully decommissioned
  (absent from Groq's live `/v1/models`). Replaced with four models confirmed working via real
  calls: Gemini → `gemini-3.1-flash-lite` + `gemini-3.5-flash`; Groq → kept
  `llama-3.1-8b-instant` (still valid) and replaced the dead 70B entry with
  `llama-3.3-70b-versatile` (Groq's own documented successor). Also updated the two README
  example-table rows (`README.md`, `client/README.md`) that cited the decommissioned
  `llama3-70b-8192`. Verified end-to-end: booted `uvicorn`, confirmed `GET /llm/gemini` and
  `GET /llm/groq` serve the new lists, uploaded a real PDF for both providers, then hit
  `POST /chat` for all four provider/model combinations and got correct, grounded answers
  (not mocked) for each.
- **Files touched:** `server/config/settings.py`, `README.md`, `client/README.md`,
  `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — chat model name fix, no retrieval logic or ranking
  changed
- **Next:** Continue Phase 0: get the Streamlit frontend running end-to-end, decide and
  shortlist the corpus. Given two provider-model lists have now gone stale within one
  session, consider whether Phase 7 observability should include a startup smoke-test
  against configured models rather than discovering staleness via user-facing errors.

### 2026-08-05
- **Done:** Locked the corpus domain: research papers on Retrieval-Augmented Generation / efficient
  LLM retrieval methods, sourced via the arXiv API. Chosen because it's fully scriptable to source
  (no manual collection, no provenance ambiguity to resolve by hand), forces every phase to matter on
  its own (exact method/benchmark names need hybrid search, citation relationships give GraphRAG a
  structural backbone, benchmark tables/figures need multimodal parsing), and is a memorable, slightly
  recursive pitch — a RAG system that helps research RAG techniques. Scope: ~40-50 papers via arXiv
  API (`cs.CL`/`cs.IR`/`cs.LG`, last ~3 years) plus a small seeded set of older foundational papers
  (DPR, RAG, cross-encoder reranking, GraphRAG) so the citation graph has real internal connectivity
  instead of references mostly pointing outside the corpus. Citation edges will come from the
  Semantic Scholar Graph API (structured — arXiv's own metadata has no reference list); Phase 4's
  actual differentiator will be an LLM-extracted semantic layer (`evaluates_on`/`extends`/`outperforms`)
  on top of that structured `cites` backbone, since a pure citation graph is just a metadata lookup and
  doesn't by itself justify GraphRAG. Wrote the domain, sourcing plan, and these decisions into
  CLAUDE.md and PROJECT_LOG.md.
- **Files touched:** `CLAUDE.md`, `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — domain/corpus decision, no retrieval logic touched
- **Next:** Build the arXiv + Semantic Scholar intake script, run it, and produce a candidate paper
  list (~40-50 + anchors) for review. Check internal citation density before finalizing. Freeze the
  corpus before Phase 1's eval baseline runs.

### 2026-08-05 (2)
- **Done:** Built `scripts/corpus_intake/fetch_candidate_papers.py` — pulls recent arXiv candidates
  (`cs.CL`/`cs.IR`/`cs.LG`, keyword-filtered, last 3 years) plus the seeded anchor-paper list, and
  checks internal citation density via the Semantic Scholar Graph API before anything gets frozen.
  Smoke-tested successfully (correct titles resolved for both search results and anchors — DPR,
  REALM, ColBERT, RAG, Self-RAG, GraphRAG, Lost in the Middle, Passage Re-ranking with BERT all
  confirmed correct). First full run (~80 recent + 8 anchors, citation lookups on) crashed on a
  Semantic Scholar read-timeout — root cause: the script was also fetching each paper's full
  `citations` list (who cites it), which is unbounded and huge for a highly-cited foundational paper.
  Fixed by only fetching `references` (a paper's own bibliography, always small), since an internal
  edge "A cites B" is fully detectable from A's own reference list — no need for the reverse lookup.
  Added retry/backoff on network errors too, since one flaky request shouldn't kill the whole run.
  Reran with the fix; rerun was in progress in the background at end of session.
- **Files touched:** `scripts/corpus_intake/fetch_candidate_papers.py` (new), `data/papers/candidates.json`
  (new), `data/papers/candidates.md` (new)
- **Eval numbers (if changed):** N/A — corpus tooling, no retrieval logic touched
- **Next:** Confirm the fixed run completes and produces a real candidate list with sane internal
  citation density; review and freeze the corpus before Phase 1's eval baseline.

### 2026-08-05 (3)
- **Done:** Cleaned all references to an earlier, discarded corpus domain out of CLAUDE.md and
  PROJECT_LOG.md at the user's request, so only the current research-paper domain and plan remain
  in both live docs. Removed the old corpus shortlist section, the Key Architecture Decisions rows
  tied to it, and the corresponding session log entries; kept the domain-neutral engineering work
  (dependency consolidation, LangChain import fixes, provider model fixes) since none of that was
  domain-specific. Old content remains in git history only, not in either live doc.
- **Files touched:** `CLAUDE.md`, `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — documentation cleanup only
- **Next:** Confirm the candidate-paper intake run completed cleanly, review the list (checking
  internal citation density), freeze the corpus, then start Phase 1.

### 2026-08-05 (4)
- **Done:** Closed out Phase 0. The Semantic Scholar per-paper lookup was still failing under real
  load (two duplicate script instances had been running at once, hammering the unauthenticated
  per-paper endpoint into near-total rate-limiting); root-caused and fixed by switching to S2's batch
  endpoint (one POST for the whole set instead of ~88 sequential GETs) — full run then finished in
  under 3 minutes. Real citation data showed the 8 original anchor papers were already densely
  interconnected (RAG alone: 26 internal edges) — the "isolated node" problem an earlier external
  review had flagged, and its proposed fix (7 new "bridge" papers), turned out to be based on a
  smoke-test run that had citation lookups turned off entirely, not on real disconnection. Verified
  the 7 proposed bridge papers individually against the live arXiv API before trusting any of them:
  6 checked out and were added as anchors (RAGAS, ColBERTv2, M3-Embedding, Adaptive-RAG, CRAG,
  ColPali); the 7th was a fabricated citation — the claimed ID resolved to a completely unrelated
  paper and was rejected. Ran the full pipeline (94 candidates: 80 recent + 14 anchors, 141 internal
  edges), then curated it down with an explicit, reviewable keep-list rather than a pure edge-count
  cutoff — cross-checked against actual abstracts, not just titles, which caught one paper
  (X-KGRank) that read as on-topic by title alone but was actually a recommender-systems paper by
  abstract. Final curated set: 45 papers, 112 internal edges. Built `freeze_corpus.py`, downloaded
  all 45 PDFs (one initial 404 traced to a stale versioned URL in the arXiv feed, fixed by always
  requesting the unversioned/latest URL), and wrote `data/papers/corpus_manifest.json` (arXiv ID,
  authors, category, source URL, sha256, download timestamp, citation count per paper) — 89.0 MB
  total, 43/45 found in Semantic Scholar's index. Corpus is now frozen. Also rewrote the "Corpus
  source & scope" section and added 4 new Key Architecture Decisions rows so the reasoning behind
  the S2 batch fix, the rejected bridge-paper fix, and the abstract-based curation methodology are
  captured as interview material, not just left implicit in chat history.
- **Files touched:** `scripts/corpus_intake/fetch_candidate_papers.py`, `scripts/corpus_intake/curate_candidates.py`
  (new), `scripts/corpus_intake/freeze_corpus.py` (new), `data/papers/candidates.json`,
  `data/papers/candidates.md`, `data/papers/curated_candidates.json` (new), `data/papers/curated_candidates.md`
  (new), `data/papers/corpus_manifest.json` (new), `data/papers/raw/*.pdf` (45 new files), `PROJECT_LOG.md`
- **Eval numbers (if changed):** N/A — corpus is now frozen; no eval has been run against it yet
- **Next:** Start Phase 1 — build the 75-100 question golden set against the frozen 45-paper corpus,
  implement MRR/Recall@k/nDCG@k, and record the naive dense-only baseline in the Eval Results table.

### 2026-08-05 (5)
- **Done:** Ingested the frozen 45-paper corpus into the vector store — the prerequisite step Phase 1
  needed before a baseline eval is even possible. Wrote `scripts/ingest_corpus.py`, which reuses the
  existing `document_processor.py` (`load_documents_from_paths`, `split_documents_to_chunks`) →
  `vector_database.py` (`get_embeddings`, Chroma upsert) pipeline directly rather than duplicating it —
  only the FastAPI-upload-specific `save_uploaded_file`/`validate_pdf` step is skipped, since the corpus
  PDFs already live on disk with provenance in `corpus_manifest.json`. Caught two problems before
  trusting the result:
  1. **Path bug (mine):** first run wrote to `<repo root>/data/{provider}_vector_store` because the
     script ran from the repo root, but `VECTORSTORE_DIRECTORY` in `config/settings.py` is a *relative*
     path resolved against the process cwd, and the app is normally run via `cd server && uvicorn
     main:app` — so the real runtime location is `server/data/{provider}_vector_store`, a different
     directory. Caught this via the spot-check step (see below), not by inspection alone. Fixed by
     having the script `chdir` into `server/` before touching any vectorstore path, so it always writes
     to the same place the running app reads from; deleted the misplaced repo-root copy.
  2. **Stray pre-existing data:** the real `server/data/{groq,gemini}_vector_store` locations weren't
     empty — they held leftover manual test uploads from earlier Phase 0 sessions (`scratchtest.pdf`,
     `scratchtest2.pdf`, and the user's own resume in the gemini collection). Confirmed with the user
     before touching it (not something this session created); cleared both collections so the vector
     store Phase 1's eval baseline gets measured against contains only the frozen 45-paper corpus.
  With both fixed, ran groq ingestion clean: 45 PDFs → 736 page-documents → **2131 chunks**, all
  embedded (local `sentence-transformers/all-MiniLM-L12-v2`) and upserted into
  `server/data/groq_vector_store` (verified via `get_collections_count`). Spot-checked with 3
  similarity searches against known paper content — all returned correct, on-topic chunks: a DPR query
  surfaced the actual DPR paper (`2004.04906.pdf`) plus other dense-retrieval discussion; an RRF query
  surfaced a chunk directly citing Cormack/Clarke/Buettcher's Reciprocal Rank Fusion paper; a GraphRAG
  community-summarization query surfaced four chunks all from the actual GraphRAG paper
  (`2404.16130.pdf`), matching its map-reduce/community-hierarchy content. gemini ingestion attempted
  but failed immediately on the first embedding batch with `429 RESOURCE_EXHAUSTED` (Google API quota
  exceeded) — added smaller/slower batching plus resume support to the script for gemini specifically,
  but per the user, gemini ingestion is deferred to a later session rather than retried now.
- **Files touched:** `scripts/ingest_corpus.py` (new), `server/data/groq_vector_store/*` (new, 2131
  chunks), `PROJECT_LOG.md`. Cleared (not touched otherwise): stray content previously in
  `server/data/groq_vector_store` and `server/data/gemini_vector_store`.
- **Eval numbers (if changed):** N/A — ingestion only, no eval harness exists yet to run
- **Next:** Retry gemini ingestion once API quota resets (`python scripts/ingest_corpus.py --provider
  gemini` — now has smaller batches + delay + resume support built in). Otherwise continue Phase 1:
  build the golden question set, implement MRR/Recall@k/nDCG@k, and run the naive dense-only baseline
  against the now-populated groq vector store.

### 2026-08-14
- **Done:** Built the Phase 1 eval harness. Phase 1 is **not** complete — the infrastructure is, the
  measurement is not. Five things happened, in order:

  1. **Discovered chunks had no identity at all, and fixed it.** The golden set's whole premise
     (`expected_chunk_ids`) had no referent: chunks entered Chroma with random UUID4s, and
     `langchain_community` 0.4.2 discards even those when rebuilding Documents from search results
     (`_results_to_docs_and_scores` reads `documents` + `metadatas`, never `ids`). Added
     `server/core/chunk_ids.py` with deterministic `{arxiv_id}::p{page}::c{ordinal}` ids, written
     **both** as Chroma's `ids=` and into `metadata["chunk_id"]` — the first makes re-ingest an
     idempotent upsert, the second is the only copy that survives retrieval. Moved `chunk_size`/
     `chunk_overlap` into `chunk_ids.py` so `corpus_fingerprint()` fingerprints the chunking actually
     in use. Re-ingested groq with `--reset`: **2131 chunks, identical to the previous count**, which
     is the confirmation that chunking itself did not change. Verified: all 2131 ids unique and
     parseable, all 45 papers present, `metadata["chunk_id"]` agrees with the Chroma primary key on
     every row, ids survive the retrieval round-trip, re-upserting 50 existing rows leaves the count
     at 2131 (so upsert-by-id genuinely dedupes), and the three known-good queries from the
     2026-08-05 (5) session still return their correct source papers.
  2. **Server plumbing, strictly additive.** `find_similar_chunks(k=4)`, a new
     `find_similar_chunks_with_scores` using `similarity_search_with_score` (named `l2_distance`, not
     `score` — the collection is `hnsw:space=l2` and embeddings are unnormalised, so the
     `L2² = 2−2cos` identity does not hold and `similarity_search_with_relevance_scores` would emit
     negative "similarities"), plus optional `k`/`include_scores` on `/vector_store/search` and
     `include_context` on `/chat`. All defaults preserved and verified against the booted app: search
     with no args still returns exactly 4 bare Documents, `/chat` with no args still returns a bare
     answer string (the Streamlit client renders `data` directly, so an unconditional dict would have
     broken it).
  3. **Metrics core, pure stdlib, 60 tests passing in 0.16s.** Implemented the equivalence-group
     relevance model and the `RR_cov` MRR (see two new Key Architecture Decisions rows). The purity
     constraint is what makes the `all_required` path testable at all, since the bulk pilot contains
     no multi-hop questions. Five fixtures pin decisions that would otherwise drift silently:
     redundancy invariance for `any_of` (T3) and for the general case (T7), the leniency gap between
     `RR_cov`/mean/first asserted in one test (T6), IDCG truncation at `min(N,k)` (T9), and
     macro-not-micro aggregation (T18).
  4. **Golden-set generation, and the defect the pilot existed to catch.** First pilot attempt: the
     model rejected 12 of 18 sampled chunks as unusable. Rather than guess at better heuristics,
     measured the feature distribution of the rejected vs accepted chunks and fitted a lexical screen
     to it (venue-phrase count for bibliographies, alpha/digit ratio and sentence-boundary count for
     tables) — it now screens out 870 of 2131 chunks, and acceptance went from 33% to 67%. Added
     resampling so a model skip refills from the *same* paper rather than silently eroding the
     per-paper stratification. Second pilot produced 18 questions with low measured vocabulary
     leakage (question↔chunk Jaccard: median 0.052, max 0.105, none above the 0.35 flag).
     **Then the end-to-end run exposed the real problem.** Of 4 complete misses, at least 3 were
     *mislabelled gold, not retrieval failures*: b0016 asked how negatives are chosen with gold
     pointing at a related-work citation list; b0005 asked how VecTree-RAG works with gold pointing at
     the paper's **title page**, while retrieval returned that paper's Conclusion — which answers it
     exactly — and scored 0. Root cause was an asymmetry I had built in: adjacent candidate chunks had
     to pass a strict "does this passage alone answer the question?" test, while the source chunk was
     assumed to be gold by construction. Fixed so the source faces the same test, plus a named-entity
     requirement and near-duplicate rejection (three pilot questions were variants of the same
     question; two retrieved the same top-1 chunk, so one was a guaranteed miss regardless of
     retrieval quality). Regeneration is written but **has not run** — see Blocked below.
  5. **Judge built and smoke-tested live.** `eval/judge.py` scores `groundedness` (against the
     contexts actually given to the generator) separately from `answer_correctness` (against the
     reference), records `judge_model` per verdict, caches by content hash, and reports loudly if the
     judge pool ever blends. The smoke test deliberately checks *discrimination*, not liveness: it
     feeds a fabricated answer and fails the model if it does not reject it. `gemini-3.5-flash`
     **passes** (scored the fabricated answer groundedness=1). Its first run had failed with
     `TypeError: expected string or bytes-like object, got 'list'`, which looked like the quota
     exhaustion this key has a history of but was actually my own bug — `ChatGoogleGenerativeAI`
     returns `.content` as a list of parts, not a str. Worth recording as the same lesson in a new
     costume: a failure that resembles a known infra problem is still worth reading before being
     attributed to it.

- **Blocked:** Groq `llama-3.3-70b-versatile` hit its **daily** token cap (TPD 100,000; 99,880 used)
  during pilot generation. Question generation costs ~4K tokens per accepted question (chunk +
  self-verification + up to two neighbour checks), so the 18-question pilot alone needs ~100K and the
  full 75 would need ~400K. The model answers again only because a rolling window frees slivers.
  `llama-3.1-8b-instant` and both Gemini models are unaffected. This is a planning constraint, not a
  defect — the decision of which model regenerates the questions is open (see Open questions/TODOs).
- **Files touched:** `server/core/chunk_ids.py` (new), `server/core/document_processor.py`,
  `server/core/vector_database.py`, `server/api/routes.py`, `server/api/schemas.py`,
  `scripts/ingest_corpus.py`, `pyproject.toml` (pytest dev extra + config),
  `eval/{__init__,_bootstrap,metrics,golden_set,retrieval,report,run_eval,inspect_chunks,generate_questions,judge}.py`
  (all new), `eval/tests/{conftest,test_metrics,test_golden_set,test_chunk_ids}.py` (new),
  `server/data/groq_vector_store/*` (re-ingested, + `ingest_manifest.json`),
  `eval/data/pilot_draft.json` (superseded, defective gold), `eval/results/*` (harness-validation run),
  `PROJECT_LOG.md`
- **Eval numbers (if changed):** **None recorded, deliberately.** The pilot run produced MRR 0.508 /
  Recall@5 0.611 / nDCG@10 0.540 (n=18, depth 50), but against the defective golden set. Those figures
  validate that the harness runs end-to-end; they are **not** a baseline and are not in the Eval
  Results table.
- **Next:** Regenerate the pilot with the gold self-verification fix once the generator-model question
  is settled, re-run the harness, then have the user hand-score the judge-validation set. After that,
  scale to ~75 bulk questions, merge the user's ~20 hand-written hard questions, freeze
  `golden_set.v1.json`, and record the first real baseline.

### 2026-08-17
- **Done:** Regenerated the golden set at scale, then audited it before trusting it. Generated 75
  bulk questions with `llama-3.1-8b-instant` (self-verification + named-entity + dedup filters from
  the 2026-08-14 fix), sanity-checked against the live retriever (MRR 0.406, Recall@5 0.547,
  nDCG@10 0.453, n=75 — a plausible, not suspiciously perfect, 24% miss rate). Then audited by hand:
  read the full text of every RR=0 miss, a seeded 15-question sample of hits, and all 22 multi-chunk
  gold groups, adjudicating each fine/thin/mis-sourced/wrong against "does the chunk alone answer the
  question, no outside knowledge". Found and fixed a systematic defect — several `any_of` groups
  included an adjacent "alternative" chunk that didn't independently answer the question, granting
  unearned retrieval credit (two questions were scoring *only* via the non-answering member). Net:
  4 questions dropped (title-page/citation-list-as-gold repeats of the known failure pattern, plus one
  degenerate self-answering question), 2 repointed to a chunk that actually explains the asked-for
  mechanism, 7 groups tightened, 3 `expected_answer`s corrected against flattened-table
  misreadings. 75 → 71, re-scored: MRR 0.384 / Recall@5 0.507 / nDCG@10 0.427. Separately authored
  the 20 hand-written hard questions (`eval/data/hard_draft.json`): 5 paraphrase-heavy, 5 cross-paper,
  5 within-paper multi-hop, 4 table-dependent (10 `all_required`, 10 `any_of`), every gold chunk read
  in full before the question was written, no LLM calls used for drafting. Validated clean (0 errors,
  0 warnings) against the live collection, and a merged sanity run showed the `all_required` stratum
  appearing for the first time and scoring far below `any_of` (MRR 0.044 vs 0.379, n=91) — real signal,
  not noise.
- **Files touched:** `eval/data/bulk_draft.json` (75→71, edited in place), `eval/data/
  bulk_questions_review.md` (regenerated), `eval/data/hard_draft.json` (new, 20 questions),
  `eval/results/*` (recheck/audited/hardset_check runs), `PROJECT_LOG.md`
- **Eval numbers (if changed):** Still not the recorded baseline — both files remained unfrozen
  drafts pending review. Bulk audited: MRR 0.384 / Recall@5 0.507 / nDCG@10 0.427 (n=71). Merged
  draft sanity check: MRR 0.343 / Recall@5 0.478 / nDCG@10 0.408 (n=91; `any_of` 0.379/0.506/0.424,
  `all_required` 0.044/0.250/0.272).
- **Next:** Human review of both drafts (or a second independent-model pass), judge validation,
  freeze `golden_set.v1.json`, record the real baseline.

### 2026-08-20
- **Done:** Closed out the golden-set and Phase 1 checklist items still open. (1) **Second-pass
  audit:** ran the two draft files (with full gold-chunk text embedded, no vector-store access
  needed) through Gemini as an independent reviewer using the project's own fine/thin/mis-sourced/
  wrong taxonomy. Its own summary arithmetic didn't reconcile with its row-level output (a live
  instance of "don't trust a model's self-tally, recompute from the primary data" — see the new Key
  Architecture Decisions row) so every one of its 12 bulk flags was independently re-verified against
  full chunk text before any fix landed: 10 confirmed real (5 repointed — e.g. APS-RAG's "corrective
  loop" gold moved from a related-work sentence naming the components to the actual state-machine
  description; 2 dropped as unfixable in-corpus, including a Contriever question with no standalone
  Contriever paper in this 45-paper corpus; 3 question/answer wording fixes), 2 rejected as
  over-strict. The 20 hard questions came back clean. Bulk: 71 → 69, re-scored MRR 0.391 / Recall@5
  0.522 / nDCG@10 0.441. (2) **Fixed a live model-drift bug found while building judge-validation
  material:** both configured Groq chat models (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
  are 404 `model_not_found` — decommissioned, not a quota issue — confirmed via a live
  `client.models.list()` call, which also meant the deployed app's `/chat` endpoint was broken for
  the `groq` provider. Verified `openai/gpt-oss-120b`/`openai/gpt-oss-20b` live with real `invoke()`
  calls (a candidate, `qwen/qwen3.6-27b`, was rejected — it leaks its `<think>` trace into
  `.content`) and swapped them into `server/config/settings.py`, `eval/judge.py`'s fallback, and
  `eval/generate_questions.py`'s generator, plus the two stale README example rows. (3) **Judge
  validation set:** built 20 items (15 clean + 5 deliberately generated against mismatched context)
  using the now-live `openai/gpt-oss-120b` as generator. The first draft of mismatched-context pairs
  used cross-topic donors and got refused ("I don't know") on all 5 — a refusal scores groundedness=5
  and defeats the stress test — so re-picked same-shape, same-register donors (another system's own
  mechanism description) that the model pattern-matched onto and confidently misattributed instead.
  Sent to an independent reviewer (Gemini): all 5 planted items correctly scored groundedness=1, plus
  3 "should-be-clean" items were correctly flagged for real generator hallucination the setup hadn't
  even engineered for — a clean discrimination result. Comparing these scores against the actual
  production judge's own verdicts on the same 20 items is still open — `gemini-3.5-flash` hit its
  20-req/day free-tier cap mid-run (a clean run needs 21 calls including the smoke test, which
  structurally can't fit in one day on this tier regardless of retries); `eval/judge.py`'s
  `validation_report()` is ready once quota resets. (4) **Froze `eval/data/golden_set.v1.json`**
  (69 bulk + 20 hard = 89 questions, validated clean) and **recorded the first real Phase 1
  baseline.**
- **Files touched:** `eval/data/bulk_draft.json` (71→69, edited in place), `eval/data/
  bulk_questions_review.md` (regenerated), `eval/data/golden_set.v1.json` (new, frozen),
  `server/config/settings.py`, `eval/judge.py`, `eval/generate_questions.py`, `README.md`,
  `client/README.md`, `eval/results/*` (bulk_gemini_audited, naive_dense_baseline runs),
  `PROJECT_LOG.md`
- **Eval numbers (if changed):** **First real baseline recorded** — `naive_dense_baseline` against
  `golden_set.v1.json` (v1, n=89): MRR 0.347 / Recall@5 0.489 / nDCG@10 0.418 (pooled). By stratum:
  `any_of` (n=79) MRR 0.385 / Recall@5 0.519 / nDCG@10 0.436; `all_required` (n=10) MRR 0.044 /
  Recall@5 0.250 / Sufficiency@5 0.100 / nDCG@10 0.272 — nine of ten multi-hop questions are
  unanswerable by naive dense retrieval within a stuffed-context chain's usable depth, exactly the
  headroom Phase 4 exists to close.
- **Next:** Retry judge validation against the real `gemini-3.5-flash` once its daily quota resets,
  compute the human/reviewer-vs-judge agreement via `validation_report()` (kappa, rejection recall),
  and record that outcome here. Then Phase 2 — hybrid retrieval (dense + BM25 + RRF), re-scored
  against `golden_set.v1.json` so the `naive_dense_baseline` row above is the real comparison point.
  Gemini vectorstore ingestion is still outstanding (unrelated, see Open questions/TODOs).

### 2026-08-20 (2)
- **Done:** Quota reset; retried the judge-validation comparison this entry's "Next" left open.
  Reran `eval/judge.py` against the same 20-item bundle — smoke test passed (correctly scored a
  fabricated answer groundedness=1), all 20 items scored by the real production judge
  (`gemini-3.5-flash`, `fallback_count=0`, one transient API error on item 20 that succeeded on
  retry). Computed agreement between the reviewer's earlier scores and the judge's via
  `validation_report()`. **The discrimination check — the actual point of this exercise — passes
  cleanly:** all 5 deliberately-unfaithful items (generated against mismatched context) were scored
  groundedness=1 by *both* scorers, independently, with zero disagreement; `rejection_recall=1.0` in
  both directions on both metrics (everything either scorer rejected at ≤2, the other rejected too).
  Overall agreement was moderate on groundedness (kappa 0.30, exact 45%, within-1 75%) and strong on
  answer_correctness (kappa 0.69, exact 80%, within-1 95%) — inspected all 5 groundedness
  disagreements individually rather than accepting the kappa number at face value, since a
  moderate-agreement number alone doesn't say *which direction* the disagreement runs, and that
  direction is the part that actually matters here. Every disagreement was the production judge
  scoring **lower** than the reviewer, and every one of its cited `unsupported_claims` checked out
  as real: the generator (`openai/gpt-oss-120b`) has a habit of padding an otherwise-correct core
  answer with fabricated specifics — invented equations, similarity thresholds, benchmark rows,
  step-by-step pipelines — that aren't in the context at all, and the production judge caught this
  consistently while the reviewer scored the same answers more leniently. Net conclusion: the
  production judge is validated, and where it diverges from a second-model reviewer, it diverges
  toward being *more* careful, not less — the safer failure mode for an eval harness to have. Closes
  the last open Phase 1 checklist item; **Phase 1 is now fully done.**
- **Files touched:** `eval/judge.py` (no code change, just invoked), `PROJECT_LOG.md`. Scratch
  artifacts only in the session's temp scratchpad dir (`production_judge_verdicts.json`,
  `judge_validation_final_report.json`) — not part of the repo.
- **Eval numbers (if changed):** Not a retrieval number — this is the judge-validation result
  itself: groundedness kappa 0.30 (rejection_recall 1.0, 5/20 disagreements, all judge-stricter and
  all individually verified as the judge being right), answer_correctness kappa 0.69
  (rejection_recall 1.0, 1/20 disagreement). Recorded in the Phase 1 status board above and the new
  Key Architecture Decisions row.
- **Next:** Phase 2 — hybrid retrieval (dense + BM25 + RRF), re-scored against `golden_set.v1.json`.
  Separately worth a look before Phase 2 gets deep: `openai/gpt-oss-120b`'s fabricated-elaboration
  habit surfaced here is a real generation-quality issue (not a retrieval one), independent of
  whatever Phase 2 changes about retrieval — noting it so it doesn't get mistaken for a Phase 2
  regression later if `answer_correctness`/groundedness numbers move at generation time.

### 2026-08-20 (3)

- **Done:** Phase 2 — hybrid retrieval, built, verified, measured, and shipped as the default.
  1. **Sparse arm, hand-rolled and cross-checked.** `server/core/bm25.py` (pure stdlib, ~250 lines:
     tokenizer, inverted index, Lucene-IDF BM25, plus an `explain()` that returns per-term score
     contributions). Correctness is not asserted on my own say-so: `rank_bm25` was installed
     temporarily, fed the *same* tokenizer output so the comparison isolates the scoring arithmetic,
     and checked for **exact** score agreement across all 2131 chunks on 20 real golden-set queries
     — 20/20, worst |delta| 1.4e-14 — then uninstalled. This is why `BM25Index` carries an
     `idf_variant` switch: the shipped `"lucene"` IDF is strictly positive, while `"okapi"`
     reproduces rank_bm25's negative-IDF-with-epsilon-floor formula exactly, which is what turns the
     cross-check into an equality assertion instead of an eyeball. The shipped variant's top-10
     overlaps the reference 8.7/10 on average — divergence expected, since a different IDF formula
     that produced identical rankings would mean the switch does nothing.
  2. **Tokenization decided explicitly, from measurements rather than instinct.** Checked the corpus
     before choosing: 212 of 2131 chunks (~10%) contain PDF ligature glyphs, so NFKD normalization
     is load-bearing, not hygiene — without it "identification" splits into `identi` + `cation`
     across a tenth of the corpus. Hyphenated compounds are pervasive (`retrieval-augmented` ×380,
     `top-k` ×94, `re-ranking` ×92), so compounds are indexed *and* sub-tokenized. No stemming and no
     stopword removal, on the principle that BM25 is in the pipeline precisely to catch the exact
     terms a bi-encoder blurs, and an aggressive tokenizer spends that advantage. All of it pinned as
     behaviour in `eval/tests/test_bm25.py` and versioned by `TOKENIZATION_VERSION`.
  3. **Fusion.** `server/core/hybrid_retriever.py`: unweighted RRF at `k=60` (Cormack et al. 2009),
     fixed a priori, over a 50-candidate pool per arm. The BM25 index is built from the Chroma
     collection's own documents rather than by re-parsing PDFs, so the arms cannot drift apart, and
     it is cached per provider and invalidated on upload (a stale sparse arm fused with a fresh dense
     arm is retrieval over two different corpora).
  4. **Pre-registered the decision rule before running anything.** Written into the Eval Results
     section first: pooled ΔnDCG@10 > 0, paired-bootstrap 95% CI lower bound > 0, no stratum
     regressing; secondary metrics reported but non-voting. `eval/compare_runs.py` applies it
     mechanically and exits non-zero if it does not fire. This finally uses `paired_delta_ci`, which
     has sat unused in `metrics.py` since Phase 1 waiting for exactly this comparison.
  5. **Result: the rule fired, decisively.** MRR 0.347 → **0.485**, Recall@5 0.489 → **0.635**,
     nDCG@10 0.418 → **0.571**; ΔnDCG@10 **+0.153, 95% CI [+0.097, +0.210]**, n=89 paired. 58
     questions up, 5 down, 23 total misses rescued and **zero** previously-found questions newly
     missed. Flipped `DEFAULT_RETRIEVAL_MODE` to `hybrid`.
  6. **The finding that beats the aggregate:** of 106 gold chunks reached, **29 (27.4%) were reached
     by the BM25 arm alone** and only 5 by dense alone — and **28 of 89 questions** had gold evidence
     that dense retrieval never surfaced *at any depth up to 50*. That is not "dense ranked it
     poorly"; it is evidence a cross-encoder could never have reranked, because reranking only
     reorders what retrieval already found. It also says the two arms are genuinely complementary
     rather than redundant.
  7. **Sensitivity, deliberately not tuning.** `scripts/rrf_sensitivity_sweep.py` retrieves both arms
     once per question and re-fuses under each config (exact, and ~6x cheaper than re-running the
     harness). Pooled nDCG@10 spread across `k ∈ {10,30,60,100}` is 0.042 against a dense→hybrid
     delta of 0.153, so the conclusion does not depend on the constant. Worth stating plainly:
     `k=10` scores **+0.042 higher** than the shipped `k=60`. That number is not claimed — taking it
     would be fitting a hyperparameter to the only 89 questions the result is reported on.
  8. **Live end-to-end check, which produced the best demonstration of the phase.** Booted the app
     and asked `/chat` "What is residual compression in ColBERTv2?" in both modes against the same
     collection and model. **Dense-only** retrieved three chunks from an unrelated paper and answered
     **"I don't know."** **Hybrid** retrieved the actual ColBERTv2 paper and answered correctly. Same
     question, same LLM, retrieval the only difference. Also confirmed backward compatibility: a
     search with no arguments still returns 4 bare Documents, `/chat` with no arguments still returns
     a bare answer string (the Streamlit client renders `data` directly), and an unrecognised
     `retrieval_mode` is rejected loudly rather than falling back to dense.
  9. **Confirmed the baseline itself did not move.** Phase 2 edited `eval/retrieval.py`'s
     `RetrievedChunk` (reordered fields, made `l2_distance` optional), which the dense baseline also
     flows through — so a silent perturbation there would have invalidated the entire comparison
     while leaving both runs looking healthy. Re-ran the dense path end-to-end
     (`dense_regression_check`) and diffed it against the recorded `naive_dense_baseline`
     per-question: **89/89 identical rr, nDCG and top-10 rankings**, pooled figures identical to four
     decimals. The +0.153 is attributable to the fusion, not to harness drift.

- **Files touched:** `server/core/bm25.py` (new), `server/core/hybrid_retriever.py` (new),
  `eval/compare_runs.py` (new), `scripts/verify_bm25_against_rank_bm25.py` (new),
  `scripts/rrf_sensitivity_sweep.py` (new), `eval/tests/{test_bm25,test_rrf}.py` (new, 45 tests),
  `server/core/vector_database.py`, `server/core/llm_chain_factory.py`, `server/config/settings.py`,
  `server/api/{routes,schemas}.py`, `eval/{retrieval,run_eval,inspect_chunks}.py`,
  `eval/results/{20260820T165153Z_hybrid_rrf60,20260820T165344Z_rrf_sensitivity_sweep,20260820T165806Z_dense_regression_check}/*` (new),
  `PROJECT_LOG.md`, `CLAUDE.md`. No new runtime dependency was added — that was the point.
- **Eval numbers (if changed):** `hybrid_rrf60` vs `naive_dense_baseline`, both against
  `golden_set.v1.json` (n=89): pooled MRR 0.347 → 0.485, Recall@5 0.489 → 0.635, nDCG@10 0.418 →
  0.571. By stratum: `any_of` (n=79) 0.385 → 0.534 / 0.519 → 0.684 / 0.436 → 0.598; `all_required`
  (n=10) 0.044 → 0.095 / 0.250 → 0.250 / 0.272 → 0.357, with **Sufficiency@5 flat at 0.100, CI
  exactly [0, 0]** — hybrid gets closer on multi-hop without making it answerable. Full paired
  intervals and the arm-attribution table are in the Eval Results section. Per the note recorded
  there before the run: these are a **lower bound**, since the golden set still only labels chunks
  dense-only retrieval surfaced.
- **Next:** Phase 3 — cross-encoder reranking (BGE-reranker-v2-m3) over the hybrid top-50, re-scored
  against `golden_set.v1.json` with `compare_runs.py` against **both** prior rows. Two things to
  carry in: (a) the re-pooling protocol executes once Phase 3 exists — union of chunks surfaced by
  dense / hybrid / hybrid+rerank, judge only the additions, bump to v2, re-score every prior row;
  (b) the 27.4% BM25-only figure sets a realistic ceiling on what reranking alone can add, since a
  reranker cannot recover what retrieval never returned.

---

## Open questions / TODOs

### Phase 1 — closed out

Judge validation completed 2026-08-20 (2) — see the Phase 1 status board and that session log entry.
Nothing left open from Phase 1 itself.

### Standing constraints for later phases

- **`gemini-3.5-flash` daily quota is a real, recurring constraint** — 20 requests/day on this key's
  free tier (confirmed twice this session: it blocked the first judge-validation attempt outright,
  and a clean 20-item run plus its smoke test is 21 calls — already over the cap on a first try even
  with nothing wasted). Any future batch judge run (a full eval pass with groundedness scoring at
  scale, not just retrieval) needs to either be sized to fit under that cap, split across days, or the
  key upgraded off the free tier before Phase 2/3 comparisons that need judge scores at scale.
- **`openai/gpt-oss-120b` (the current Groq generator) fabricates elaboration even on correctly-
  grounded answers** — invented equations, thresholds, benchmark rows, step-by-step pipelines not in
  the context, on otherwise-correct answers. Surfaced during judge validation (2026-08-20 (2)), not
  yet characterized at scale. Worth watching once real end-to-end `/chat` answers are being generated
  and scored in later phases — the production judge does catch it (that's confirmed), but it means
  `answer_correctness` can be high while `groundedness` is mediocre on a meaningful fraction of
  answers, independent of anything Phase 2/3 changes about retrieval.

- **The BM25 index is in-memory, per process, rebuilt on startup** (~0.7s for 2131 chunks: 0.51s to
  read the collection + 0.12s to tokenize). Fine at this corpus size and deliberately so — it is one
  fewer moving part than a persisted sparse index, and it cannot drift from the dense arm. It is also
  the honest answer to "what breaks at 10x the corpus": at ~20k chunks the startup cost and the
  per-worker memory duplication start to matter, and the fix is a persisted/served sparse index
  (Tantivy, OpenSearch, or Chroma's own full-text support) rather than a bigger in-process dict.
  Revisit in Phase 7 alongside the other serving work, not before — it changes zero rankings.
- **`reset_hybrid_cache()` must be called after anything that writes to a collection.** It is wired
  into `upsert_vectorstore_from_pdfs` now; any *new* write path (Phase 5 re-ingestion especially)
  must call it too, or the sparse arm silently serves a stale corpus while the dense arm serves the
  fresh one.

### Deferred deliberately (do NOT fold into another change silently)

- **Unify the k mismatch.** `find_similar_chunks` defaults to k=4, `build_llm_chain` retrieves k=3, so
  the search endpoint and the chat path do not see the same set. Left alone on purpose: this is a
  *ranking* change, and making it before the baseline exists would bake the fix into the reference
  point and destroy any clean before/after for it. Do it as its own measured change after Phase 1.
- **`get_embeddings` reloads MiniLM from disk on every call**, and `/chat` rebuilds the whole chain per
  request. Real serving problem, no effect on ranking. Defer to Phase 7 alongside the other serving
  work. (The eval harness already sidesteps it by constructing the retriever once.)
- **Pin `pypdf` / `langchain-text-splitters` / `chromadb` / `sentence-transformers`.** `pyproject.toml`
  pins nothing, so a transparent upgrade could re-segment the corpus and invalidate every gold chunk id
  while every run still completes. Partially mitigated now: `ingest_manifest.json` and each run's
  `run_meta.json` record library versions, and per-chunk `text_sha8` makes drift a hard validation
  error rather than a silent one. Actual pinning still outstanding.

### Re-pooling protocol (committed 2026-08-14, executes in Phase 2/3)

The Phase 1 golden set is built WITHOUT pooling candidates from the retriever under test, to avoid
biasing the eval toward naive dense retrieval. The consequence is that some legitimately-correct
chunks are unlabelled, so a later system can be penalised for finding a different-but-valid chunk.
The fix, to run once Phases 2 and 3 exist:

1. Take the **union** of chunks surfaced across the golden set by naive dense, hybrid, and
   hybrid+rerank.
2. Independently judge **only the new additions** — do not re-litigate existing labels.
3. Bump the golden set to `v2`.
4. **Re-score every prior variant against v2** and update its Eval Results row (per the comparability
   rule in that section). Pooling from three systems instead of one also materially reduces the
   single-system pooling bias, which is the standard TREC argument.

### Other

- gemini vectorstore ingestion is still outstanding — blocked on `429 RESOURCE_EXHAUSTED` (Google API
  quota) on 2026-08-05. `scripts/ingest_corpus.py --provider gemini` is ready to retry once quota/
  billing is sorted (it now batches smaller + delays + resumes from wherever it left off). groq is
  done (2131 chunks, re-ingested with deterministic ids 2026-08-14). Note the eval harness aborts
  cleanly on an empty collection, so `--provider gemini` fails loudly rather than reporting 0.000.
- `pyproject.toml` still says `description = "...financial document intelligence"` and the repo
  directory is `financial-rag-assistant` — both are leftovers from the discarded pre-2026-08-05 corpus
  domain. PROJECT_LOG's Project overview also still has **Name: TBD**, gated on the corpus being
  frozen, which it now is. Worth settling.
