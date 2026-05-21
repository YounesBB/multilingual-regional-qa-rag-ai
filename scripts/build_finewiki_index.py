"""Build E5 FAISS indexes over FineWiki chunks for cs, sk, and uk."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from e5_faiss_index import (
    AtomicJsonlWriter,
    E5Embedder,
    chunk_to_metadata,
    create_inner_product_index,
    import_faiss,
    write_faiss_index,
    atomic_write_json,
)
from full_dev_common import LANGUAGES, PROJECT_WORK
from samples.finewiki_sample import CONFIG_BY_LANG, DATASET_NAME
from smoke_rag import Chunk, chunk_document


def iter_finewiki_records(
    lang: str,
    config: str,
    split: str,
    cache_dir: Path | None,
    limit_docs: int,
) -> Iterable[dict]:
    dataset = load_dataset(
        DATASET_NAME,
        name=config,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    for index, record in enumerate(dataset):
        if limit_docs > 0 and index >= limit_docs:
            break
        item = dict(record)
        item["lang"] = lang
        yield item


def output_paths(output_dir: Path, index_tag: str, lang: str) -> dict[str, Path]:
    prefix = output_dir / f"{index_tag}_{lang}"
    return {
        "index": prefix.with_suffix(".faiss"),
        "metadata": output_dir / f"{index_tag}_{lang}_metadata.jsonl",
        "summary": output_dir / f"{index_tag}_{lang}_summary.json",
    }


def should_skip(paths: dict[str, Path], force: bool) -> bool:
    if force:
        return False
    return all(path.exists() for path in paths.values())


def flush_chunk_batch(
    chunks: list[Chunk],
    embedder: E5Embedder,
    index,
    metadata_writer: AtomicJsonlWriter,
    batch_size: int,
    counters: Counter,
) -> None:
    if not chunks:
        return
    embeddings = embedder.encode_passages(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
    )
    if index.ntotal == 0:
        if embeddings.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index {index.d}"
            )
    index.add(embeddings)
    for chunk in chunks:
        metadata_writer.write(chunk_to_metadata(chunk))
    counters["chunks"] += len(chunks)
    chunks.clear()


def build_language_index(
    lang: str,
    config: str,
    args: argparse.Namespace,
    embedder: E5Embedder,
) -> dict:
    paths = output_paths(args.output_dir, args.index_tag, lang)
    if should_skip(paths, args.force):
        print(f"[{lang}] Existing index files found; skipping. Use --force to rebuild.")
        return {
            "lang": lang,
            "skipped": True,
            "index_path": str(paths["index"]),
            "metadata_path": str(paths["metadata"]),
            "summary_path": str(paths["summary"]),
        }

    started = time.time()
    counters: Counter = Counter()
    chunk_batch: list[Chunk] = []
    index = create_inner_product_index(args.embedding_dim)
    metadata_writer = AtomicJsonlWriter(paths["metadata"])

    try:
        for record in iter_finewiki_records(
            lang=lang,
            config=config,
            split=args.split,
            cache_dir=args.cache_dir,
            limit_docs=args.limit_docs_per_lang,
        ):
            counters["documents"] += 1
            chunks = chunk_document(record, args.chunk_tokens, args.overlap_tokens)
            if args.limit_chunks_per_lang > 0:
                remaining = args.limit_chunks_per_lang - counters["chunks"] - len(
                    chunk_batch
                )
                if remaining <= 0:
                    break
                chunks = chunks[:remaining]
            chunk_batch.extend(chunks)

            while len(chunk_batch) >= args.index_batch_size:
                to_flush = chunk_batch[: args.index_batch_size]
                del chunk_batch[: args.index_batch_size]
                flush_chunk_batch(
                    to_flush,
                    embedder,
                    index,
                    metadata_writer,
                    args.embedding_batch_size,
                    counters,
                )

            if counters["documents"] % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"[{lang}] docs={counters['documents']} "
                    f"chunks={counters['chunks'] + len(chunk_batch)} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        flush_chunk_batch(
            chunk_batch,
            embedder,
            index,
            metadata_writer,
            args.embedding_batch_size,
            counters,
        )
        metadata_writer.publish()
        write_faiss_index(index, paths["index"])

        elapsed = time.time() - started
        summary = {
            "lang": lang,
            "config": config,
            "dataset": DATASET_NAME,
            "split": args.split,
            "model_name": args.model_name,
            "index_tag": args.index_tag,
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "embedding_dim": args.embedding_dim,
            "tokenizer_max_length": args.tokenizer_max_length,
            "limit_docs_per_lang": args.limit_docs_per_lang,
            "limit_chunks_per_lang": args.limit_chunks_per_lang,
            "document_count": counters["documents"],
            "chunk_count": counters["chunks"],
            "elapsed_seconds": elapsed,
            "index_path": str(paths["index"]),
            "metadata_path": str(paths["metadata"]),
        }
        atomic_write_json(summary, paths["summary"])
        print(
            f"[{lang}] wrote {counters['chunks']} chunks from "
            f"{counters['documents']} documents to {paths['index']}",
            flush=True,
        )
        return summary
    except Exception:
        metadata_writer.discard()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset input
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit-docs-per-lang", type=int, default=0)
    parser.add_argument("--limit-chunks-per-lang", type=int, default=0)

    # Chunking settings
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=128)

    # Embedding and indexing settings
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--index-batch-size", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")

    # Output paths
    parser.add_argument("--index-tag", default="e5_debug")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_WORK / "indexes",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.languages) - set(CONFIG_BY_LANG))
    if unknown:
        raise ValueError(f"Unknown FineWiki languages: {unknown}")
    if args.overlap_tokens >= args.chunk_tokens:
        raise ValueError("--overlap-tokens must be smaller than --chunk-tokens")
    if args.embedding_batch_size <= 0 or args.index_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")

    import_faiss()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building FineWiki indexes: languages={args.languages}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")

    embedder = E5Embedder(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir,
        max_length=args.tokenizer_max_length,
        local_files_only=args.local_files_only,
    )

    summaries = []
    for lang in args.languages:
        summary = build_language_index(
            lang=lang,
            config=CONFIG_BY_LANG[lang],
            args=args,
            embedder=embedder,
        )
        summaries.append(summary)

    global_summary_path = args.output_dir / f"{args.index_tag}_summary.json"
    atomic_write_json(
        {
            "index_tag": args.index_tag,
            "model_name": args.model_name,
            "languages": args.languages,
            "summaries": summaries,
        },
        global_summary_path,
    )
    print(f"Wrote global summary to {global_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
