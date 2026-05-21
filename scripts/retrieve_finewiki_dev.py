"""Retrieve FineWiki chunks for CUS-QA dev questions with E5 and FAISS."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from e5_faiss_index import E5Embedder, atomic_write_json, import_faiss
from full_dev_common import PROJECT_WORK, language_counts, limit_per_language, read_jsonl, write_jsonl
from smoke_rag import build_prompt


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(unquote(title).replace("_", " ").casefold().split())


def load_index(index_dir: Path, index_tag: str, lang: str):
    faiss = import_faiss()
    index_path = index_dir / f"{index_tag}_{lang}.faiss"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    return faiss.read_index(str(index_path))


def metadata_path(index_dir: Path, index_tag: str, lang: str) -> Path:
    path = index_dir / f"{index_tag}_{lang}_metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_selected_metadata(path: Path, positions: set[int]) -> dict[int, dict]:
    """Scan metadata once and keep only FAISS-selected chunk rows in memory."""

    if not positions:
        return {}
    selected = {}
    remaining = set(positions)
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in remaining:
                continue
            selected[row_index] = json.loads(line)
            remaining.remove(row_index)
            if not remaining:
                break
    if remaining:
        raise ValueError(
            f"Metadata file {path} was missing {len(remaining)} requested positions"
        )
    return selected


def retrieve_language(
    lang: str,
    questions: list[dict],
    embedder: E5Embedder,
    args: argparse.Namespace,
) -> list[dict]:
    print(f"[{lang}] loading FAISS index", flush=True)
    index = load_index(args.index_dir, args.index_tag, lang)
    print(f"[{lang}] index rows={index.ntotal}", flush=True)

    query_texts = [
        record.get("question_orig") or record.get("question") or ""
        for record in questions
    ]
    query_embeddings = embedder.encode_queries(query_texts, args.embedding_batch_size)
    scores, indices = index.search(query_embeddings, args.top_k)

    needed_positions = {
        int(position)
        for row in indices.tolist()
        for position in row
        if int(position) >= 0
    }
    print(
        f"[{lang}] loading {len(needed_positions)} selected metadata rows",
        flush=True,
    )
    metadata = load_selected_metadata(
        metadata_path(args.index_dir, args.index_tag, lang),
        needed_positions,
    )

    records = []
    for question_record, score_row, index_row in zip(
        questions,
        scores.tolist(),
        indices.tolist(),
    ):
        question = question_record.get("question_orig") or question_record.get("question")
        retrieved = []
        for rank, (score, position) in enumerate(zip(score_row, index_row), start=1):
            position = int(position)
            if position < 0:
                continue
            chunk = metadata[position]
            retrieved.append(
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

        target_title = normalize_title(question_record.get("wikititle"))
        target_title_rank = None
        for chunk in retrieved:
            if normalize_title(chunk.get("title")) == target_title:
                target_title_rank = chunk["rank"]
                break

        records.append(
            {
                "_retrieval_order": question_record.get("_retrieval_order", 0),
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
                    "retrieved_titles": [
                        normalize_title(chunk.get("title")) for chunk in retrieved
                    ],
                },
                "prompt": build_prompt(question, lang, retrieved),
                "generation": {
                    "skip_generation": True,
                    "answer": "",
                },
            }
        )

    del index
    del metadata
    gc.collect()
    print(f"[{lang}] retrieved {len(records)} questions", flush=True)
    return records


def build_summary(records: list[dict], args: argparse.Namespace) -> dict:
    by_lang = defaultdict(list)
    for record in records:
        by_lang[record["lang"]].append(record)

    language_summary = {}
    for lang, lang_records in sorted(by_lang.items()):
        top1_scores = [
            record["retrieval"]["chunks"][0]["score"]
            for record in lang_records
            if record["retrieval"]["chunks"]
        ]
        title_hits = sum(
            1
            for record in lang_records
            if record["retrieval"]["target_title_in_top_k"]
        )
        empty = sum(1 for record in lang_records if not record["retrieval"]["chunks"])
        full_top_k = sum(
            1
            for record in lang_records
            if len(record["retrieval"]["chunks"]) == args.top_k
        )
        language_summary[lang] = {
            "records": len(lang_records),
            "empty_retrievals": empty,
            "full_top_k_records": full_top_k,
            "target_title_hits": title_hits,
            "target_title_hit_rate": title_hits / len(lang_records)
            if lang_records
            else 0.0,
            "average_top1_score": sum(top1_scores) / len(top1_scores)
            if top1_scores
            else 0.0,
        }

    total_title_hits = sum(
        1 for record in records if record["retrieval"]["target_title_in_top_k"]
    )
    total_empty = sum(1 for record in records if not record["retrieval"]["chunks"])
    full_top_k_records = sum(
        1 for record in records if len(record["retrieval"]["chunks"]) == args.top_k
    )
    return {
        "questions_jsonl": str(args.questions_jsonl),
        "index_dir": str(args.index_dir),
        "index_tag": args.index_tag,
        "model_name": args.model_name,
        "top_k": args.top_k,
        "record_count": len(records),
        "language_counts": language_counts(records),
        "empty_retrieval_count": total_empty,
        "full_top_k_records": full_top_k_records,
        "short_top_k_records": len(records) - full_top_k_records,
        "target_title_hits": total_title_hits,
        "target_title_hit_rate": total_title_hits / len(records) if records else 0.0,
        "language_summary": language_summary,
    }


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
        default=PROJECT_WORK / "indexes" / "e5_full",
    )
    parser.add_argument("--index-tag", default="e5_full")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_WORK / "runs" / "e5_full_dev_retrieval_top5.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_WORK / "summaries" / "e5_full_dev_retrieval_top5_summary.json",
    )
    parser.add_argument("--limit-questions-per-lang", type=int, default=0)
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
    questions = limit_per_language(
        read_jsonl(args.questions_jsonl),
        args.limit_questions_per_lang,
    )
    questions_by_lang: dict[str, list[dict]] = defaultdict(list)
    for order, record in enumerate(questions):
        item = dict(record)
        item["_retrieval_order"] = order
        questions_by_lang[item.get("lang") or ""].append(item)

    embedder = E5Embedder(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir,
        max_length=args.tokenizer_max_length,
        local_files_only=args.local_files_only,
    )

    output_records = []
    for lang in sorted(questions_by_lang):
        if not lang:
            raise ValueError("Question record missing language")
        output_records.extend(
            retrieve_language(lang, questions_by_lang[lang], embedder, args)
        )

    output_records.sort(key=lambda record: record["_retrieval_order"])
    for record in output_records:
        record.pop("_retrieval_order", None)
    write_jsonl(output_records, args.output_jsonl)

    summary = build_summary(output_records, args)
    atomic_write_json(summary, args.summary_json)
    print(f"Wrote {len(output_records)} retrieval records to {args.output_jsonl}")
    print(f"Wrote retrieval summary to {args.summary_json}")
    print(summary)
    if summary["empty_retrieval_count"]:
        raise ValueError(f"empty_retrieval_count={summary['empty_retrieval_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
