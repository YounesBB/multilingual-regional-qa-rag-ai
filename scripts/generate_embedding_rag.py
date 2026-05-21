"""Generate Embedding RAG answers from precomputed FineWiki retrieval records."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from full_dev_common import (
    PROJECT_WORK,
    duplicate_identities,
    language_counts,
    limit_per_language,
    read_jsonl,
    record_identity,
    write_json,
    write_jsonl,
)
from smoke_rag import build_prompt, generate_answer, load_generator


def existing_identities(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record_identity(record) for record in read_jsonl(path)}


def load_prompt(record: dict) -> str:
    prompt = record.get("prompt")
    if prompt:
        return prompt
    question = record.get("question_orig") or record.get("question") or ""
    lang = record.get("lang") or ""
    chunks = (record.get("retrieval") or {}).get("chunks") or []
    return build_prompt(question, lang, chunks)


def generate_records(args: argparse.Namespace) -> list[dict]:
    records = read_jsonl(args.retrieval_jsonl)
    records = limit_per_language(records, args.limit_records_per_lang)
    if args.limit_records > 0:
        records = records[: args.limit_records]

    completed = existing_identities(args.output_jsonl) if args.resume else set()
    pending = [
        record for record in records if record_identity(record) not in completed
    ]
    if args.resume and completed:
        print(
            f"Resume enabled: {len(completed)} existing records, "
            f"{len(pending)} pending records",
            flush=True,
        )

    generator = load_generator(
        args.model_name,
        args.cache_dir,
        args.trust_remote_code,
        args.local_files_only,
    )

    output_records = []
    for index, record in enumerate(pending, start=1):
        prompt = load_prompt(record)
        answer = generate_answer(
            generator.tokenizer,
            generator.model,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.generation_top_k,
            generator.uses_processor,
        )
        output_record = dict(record)
        output_record["prompt"] = prompt
        output_record["generation"] = {
            "model_name": args.model_name,
            "source_retrieval_jsonl": str(args.retrieval_jsonl),
            "skip_generation": False,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generation_top_k": args.generation_top_k,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "answer": answer,
        }
        output_records.append(output_record)
        if index == 1 or index % args.log_every == 0:
            print(
                f"Generated {index}/{len(pending)} pending records",
                flush=True,
            )

    return output_records


def append_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict], args: argparse.Namespace, elapsed_seconds: float) -> dict:
    empty_answer_ids = [
        record_identity(record)
        for record in records
        if not ((record.get("generation") or {}).get("answer") or "").strip()
    ]
    retrieval_hits = sum(
        1 for record in records if (record.get("retrieval") or {}).get("target_title_in_top_k")
    )
    top1_scores_by_lang: dict[str, list[float]] = {}
    for record in records:
        chunks = (record.get("retrieval") or {}).get("chunks") or []
        if not chunks:
            continue
        top1_scores_by_lang.setdefault(record.get("lang") or "unknown", []).append(
            float(chunks[0].get("score") or 0.0)
        )

    lang_summary = {}
    for lang in sorted(set(record.get("lang") or "unknown" for record in records)):
        lang_records = [record for record in records if (record.get("lang") or "unknown") == lang]
        lang_hits = sum(
            1
            for record in lang_records
            if (record.get("retrieval") or {}).get("target_title_in_top_k")
        )
        scores = top1_scores_by_lang.get(lang, [])
        lang_summary[lang] = {
            "records": len(lang_records),
            "target_title_hits": lang_hits,
            "target_title_hit_rate": lang_hits / len(lang_records)
            if lang_records
            else 0.0,
            "average_top1_score": sum(scores) / len(scores) if scores else 0.0,
        }

    duplicate_ids = duplicate_identities(records)
    summary = {
        "model_name": args.model_name,
        "retrieval_jsonl": str(args.retrieval_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "record_count": len(records),
        "language_counts": language_counts(records),
        "empty_answer_count": len(empty_answer_ids),
        "empty_answer_ids": empty_answer_ids[:50],
        "duplicate_identity_count": len(duplicate_ids),
        "duplicate_identities": duplicate_ids[:50],
        "target_title_hits": retrieval_hits,
        "target_title_hit_rate": retrieval_hits / len(records) if records else 0.0,
        "language_summary": lang_summary,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_top_k": args.generation_top_k,
        "elapsed_seconds": elapsed_seconds,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-jsonl",
        type=Path,
        default=PROJECT_WORK / "runs" / "e5_full_dev_retrieval_top5.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_WORK / "runs" / "embedding_rag_qwen3_30b_a3b_top5.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_WORK / "summaries" / "embedding_rag_qwen3_30b_a3b_top5_summary.json",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_WORK / "hf_models")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-records-per-lang", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--generation-top-k", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    new_records = generate_records(args)

    if args.resume and args.output_jsonl.exists():
        append_jsonl(new_records, args.output_jsonl)
        all_records = read_jsonl(args.output_jsonl)
    else:
        write_jsonl(new_records, args.output_jsonl)
        all_records = new_records

    summary = summarize(all_records, args, time.time() - start)
    write_json(summary, args.summary_json)
    print(f"Wrote {len(new_records)} new records to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")
    print(summary)
    if summary["empty_answer_count"]:
        raise ValueError(f"empty_answer_count={summary['empty_answer_count']}")
    if summary["duplicate_identity_count"]:
        raise ValueError(f"duplicate_identity_count={summary['duplicate_identity_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
