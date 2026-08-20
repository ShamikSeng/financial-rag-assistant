"""Read-only chunk browser, for authoring and reviewing the golden set.

Exists so that no chunk id is ever typed by hand. Hand-typed ids are the single
biggest source of unreachable-gold errors, and unreachable gold silently
depresses every metric in a way that reads exactly like a retriever regression.

Usage (from repo root):
    python eval/inspect_chunks.py --arxiv-id 2004.04906
    python eval/inspect_chunks.py --arxiv-id 2004.04906 --page 2 --full
    python eval/inspect_chunks.py --chunk-id 2004.04906::p2::c1 --neighbours
    python eval/inspect_chunks.py --query "reciprocal rank fusion" --k 10
    python eval/inspect_chunks.py --stats
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Run as a script, sys.path[0] is eval/, not the repo root -- so `import eval.x`
# fails before _bootstrap has had a chance to fix anything. Entry points must
# put the repo root on the path themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval._bootstrap import bootstrap_server_context  # noqa: E402

bootstrap_server_context()

from core.chunk_ids import parse_chunk_id  # noqa: E402
from eval.retrieval import DenseChromaRetriever, HybridRRFRetriever, load_collection_index  # noqa: E402

PREVIEW_CHARS = 320


def preview(text: str, full: bool = False) -> str:
  if full:
    return text
  flat = " ".join(text.split())
  return flat[:PREVIEW_CHARS] + ("..." if len(flat) > PREVIEW_CHARS else "")


def sort_key(chunk_id: str):
  parts = parse_chunk_id(chunk_id)
  return (parts.arxiv_id, parts.page, parts.ordinal)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--provider", default="groq")
  ap.add_argument("--arxiv-id", help="show all chunks for one paper")
  ap.add_argument("--page", type=int, help="restrict to one 0-indexed page")
  ap.add_argument("--chunk-id", help="show one chunk")
  ap.add_argument("--neighbours", action="store_true",
                  help="with --chunk-id, also show the adjacent chunks it overlaps with")
  ap.add_argument("--query", help="run a similarity search and show the hits")
  ap.add_argument("--mode", choices=["dense", "hybrid", "bm25"], default="dense",
                  help="with --query: which retrieval path to inspect. 'bm25' shows the "
                       "sparse arm alone with its per-term score breakdown, which is how "
                       "you tell a real lexical match from one carried by a single junk "
                       "token the PDF extractor glued together.")
  ap.add_argument("--k", type=int, default=10)
  ap.add_argument("--full", action="store_true", help="print full chunk text")
  ap.add_argument("--stats", action="store_true", help="per-paper chunk counts")
  args = ap.parse_args(argv)

  ids, _sha_by_id, text_by_id = load_collection_index(args.provider)

  if args.stats:
    counts = Counter(parse_chunk_id(i).arxiv_id for i in ids)
    print(f"{len(counts)} papers, {len(ids)} chunks\n")
    for arxiv_id, n in counts.most_common():
      print(f"  {arxiv_id:>14}  {n:4d}")
    print(f"\n  mean {len(ids) / len(counts):.1f}  min {min(counts.values())}  max {max(counts.values())}")
    return 0

  if args.chunk_id:
    if args.chunk_id not in text_by_id:
      print(f"No such chunk: {args.chunk_id}", file=sys.stderr)
      return 1
    targets = [args.chunk_id]
    if args.neighbours:
      p = parse_chunk_id(args.chunk_id)
      for delta in (-1, 1):
        neighbour = f"{p.arxiv_id}::p{p.page}::c{p.ordinal + delta}"
        if neighbour in text_by_id:
          targets.append(neighbour)
      targets.sort(key=sort_key)
    for cid in targets:
      marker = " <-- target" if cid == args.chunk_id else ""
      print(f"\n=== {cid}{marker}\n{preview(text_by_id[cid], args.full)}")
    return 0

  if args.query:
    retriever = DenseChromaRetriever(args.provider)
    for hit in retriever.retrieve(args.query, args.k):
      print(f"\n=== rank {hit.rank}  l2={hit.l2_distance:.4f}  {hit.chunk_id}"
            f"\n{preview(hit.page_content, args.full)}")
    return 0

  if args.arxiv_id:
    selected = sorted(
      (i for i in ids if parse_chunk_id(i).arxiv_id == args.arxiv_id),
      key=sort_key,
    )
    if args.page is not None:
      selected = [i for i in selected if parse_chunk_id(i).page == args.page]
    if not selected:
      print(f"No chunks for arxiv_id={args.arxiv_id}"
            + (f" page={args.page}" if args.page is not None else ""), file=sys.stderr)
      return 1
    print(f"{len(selected)} chunks (page numbers are 0-indexed)")
    for cid in selected:
      print(f"\n=== {cid}\n{preview(text_by_id[cid], args.full)}")
    return 0

  ap.print_help()
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
