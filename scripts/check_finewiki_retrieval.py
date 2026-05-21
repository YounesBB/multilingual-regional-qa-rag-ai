"""Run retrieval-only checks against a FineWiki E5 FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote

from e5_faiss_index import E5Embedder, import_faiss
from full_dev_common import PROJECT_WORK, limit_per_language, read_jsonl, write_jsonl
from smoke_rag import build_prompt


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_language_indexes(index_dir: Path, index_tag: str, languages: list[str]) -> dict:
    faiss = import_faiss()
    loaded = {}
    for lang in languages:
        index_path = index_dir / f"{index_tag}_{lang}.faiss"
        metadata_path = index_dir / f"{index_tag}_{lang}_metadata.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        index = faiss.read_index(str(index_path))
        metadata = load_metadata(metadata_path)
        if index.ntotal != len(metadata):
            raise ValueError(
                f"{lang} index rows ({index.ntotal}) do not match metadata rows "
                f"({len(metadata)})"
            )
        loaded[lang] = {"index": index, "metadata": metadata}
    return loaded


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    title = unquote(title).replace("_", " ")
    return " ".join(title.casefold().split())


def retrieve_question(
    question: str,
    lang: str,
    embedder: E5Embedder,
    index_data: dict,
    top_k: int,
    batch_size: int,
) -> list[dict]:
    query_embedding = embedder.encode_queries([question], batch_size=batch_size)
    scores, indices = index_data["index"].search(query_embedding, top_k)
    metadata = index_data["metadata"]

    results = []
    for rank, (score, index_position) in enumerate(
        zip(scores[0].tolist(), indices[0].tolist()),
        start=1,
    ):
        if index_position < 0:
            continue
        chunk = metadata[index_position]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "lang": chunk.get("lang") or lang,
                "text": chunk.get("text"),
            }
        )
    return results


def build_records(args: argparse.Namespace) -> list[dict]:
    questions = limit_per_language(
        read_jsonl(args.questions_jsonl),
        args.limit_questions_per_lang,
    )
    languages = sorted({record.get("lang") for record in questions if record.get("lang")})
    indexes = load_language_indexes(args.index_dir, args.index_tag, languages)
    embedder = E5Embedder(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir,
        max_length=args.tokenizer_max_length,
        local_files_only=args.local_files_only,
    )

    records = []
    for question_record in questions:
        lang = question_record.get("lang") or ""
        question = question_record.get("question_orig") or question_record.get("question")
        retrieved = retrieve_question(
            question=question,
            lang=lang,
            embedder=embedder,
            index_data=indexes[lang],
            top_k=args.top_k,
            batch_size=args.embedding_batch_size,
        )
        target_title = normalize_title(question_record.get("wikititle"))
        retrieved_titles = [normalize_title(chunk.get("title")) for chunk in retrieved]
        target_title_rank = None
        for chunk in retrieved:
            if normalize_title(chunk.get("title")) == target_title:
                target_title_rank = chunk["rank"]
                break

        records.append(
            {
                "id": question_record.get("id"),
                "lang": lang,
                "question": question,
                "reference_answer": question_record.get("answer_orig"),
                "wikititle": question_record.get("wikititle"),
                "wiki_url": question_record.get("wiki_url"),
                "retrieval": {
                    "method": "e5_faiss",
                    "index_tag": args.index_tag,
                    "top_k": args.top_k,
                    "chunks": retrieved,
                    "non_empty": bool(retrieved),
                    "target_title_in_top_k": target_title_rank is not None,
                    "target_title_rank": target_title_rank,
                    "retrieved_titles": retrieved_titles,
                },
                "prompt": build_prompt(question, lang, retrieved),
                "generation": {
                    "skip_generation": True,
                    "answer": "",
                },
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-jsonl",
        type=Path,
        default=PROJECT_WORK / "inputs" / "cusqa_dev_all.jsonl",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_WORK / "indexes" / "e5_debug",
    )
    parser.add_argument("--index-tag", default="e5_debug")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/runs/e5_debug_retrieval_check.jsonl"),
    )
    parser.add_argument("--limit-questions-per-lang", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=PROJECT_WORK / "hf_models",
    )
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_records(args)
    write_jsonl(records, args.output_jsonl)

    non_empty = sum(1 for record in records if record["retrieval"]["non_empty"])
    title_hits = sum(
        1 for record in records if record["retrieval"]["target_title_in_top_k"]
    )
    print(
        f"Wrote {len(records)} retrieval records to {args.output_jsonl}; "
        f"non_empty={non_empty}; target_title_hits={title_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
