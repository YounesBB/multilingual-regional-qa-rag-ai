"""Evaluate retrieval evidence quality for CUS-QA dev retrieval JSONL files.

The main retrieval metric is gold-page Recall@k: whether any retrieved chunk
comes from the development example's known Wikipedia source page.  We also
compute MRR@k for the first gold-page chunk and conservative answer-string
recall proxies.  Answer-string recall is a lower-bound diagnostic, not a proof
of answer evidence absence, because valid evidence may be inflected,
paraphrased, abbreviated, or split across chunks.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from full_dev_common import PROJECT_WORK, language_counts, read_jsonl, write_json


DEFAULT_RETRIEVAL_FILES = [
    "e5_full_dev_retrieval_top5.jsonl",
    "e5_full_dev_retrieval_top10.jsonl",
]


def normalize_for_match(text: str, strip_diacritics: bool = False) -> str:
    """Normalize text for transparent string/token evidence checks."""

    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if strip_diacritics:
        decomposed = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in decomposed if not unicodedata.combining(char))
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(char)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def contains_normalized_phrase(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def token_overlap_hit(reference: str, chunk: str, threshold: float) -> bool:
    reference_tokens = normalize_for_match(reference, strip_diacritics=True).split()
    chunk_tokens = set(normalize_for_match(chunk, strip_diacritics=True).split())
    if not reference_tokens or not chunk_tokens:
        return False

    unique_reference_tokens = list(dict.fromkeys(reference_tokens))
    if len(unique_reference_tokens) <= 2:
        required = len(unique_reference_tokens)
    else:
        required = math.ceil(threshold * len(unique_reference_tokens))
    overlap = sum(1 for token in unique_reference_tokens if token in chunk_tokens)
    return overlap >= required


def reference_answer(record: dict) -> str:
    return str(record.get("reference_answer") or record.get("answer_orig") or "").strip()


def retrieved_chunks(record: dict) -> list[dict]:
    retrieval = record.get("retrieval") or {}
    chunks = retrieval.get("chunks") or []
    return chunks if isinstance(chunks, list) else []


def top_k_for_records(records: Sequence[dict], fallback: int | None = None) -> int:
    for record in records:
        retrieval = record.get("retrieval") or {}
        top_k = retrieval.get("top_k")
        if isinstance(top_k, int):
            return top_k
        chunks = retrieved_chunks(record)
        if chunks:
            return max(int(chunk.get("rank") or 0) for chunk in chunks)
    return fallback or 0


def evidence_flags(record: dict, token_threshold: float) -> dict:
    reference = reference_answer(record)
    strict_reference = normalize_for_match(reference)
    strict_hit_rank = None
    relaxed_hit_rank = None

    for chunk in retrieved_chunks(record):
        rank = int(chunk.get("rank") or 0)
        chunk_text = str(chunk.get("text") or "")
        strict_chunk = normalize_for_match(chunk_text)
        if strict_hit_rank is None and contains_normalized_phrase(
            strict_reference, strict_chunk
        ):
            strict_hit_rank = rank
        if relaxed_hit_rank is None and token_overlap_hit(
            reference, chunk_text, token_threshold
        ):
            relaxed_hit_rank = rank
        if strict_hit_rank is not None and relaxed_hit_rank is not None:
            break

    return {
        "strict_answer_string_hit": strict_hit_rank is not None,
        "strict_answer_string_rank": strict_hit_rank,
        "relaxed_answer_token_hit": relaxed_hit_rank is not None,
        "relaxed_answer_token_rank": relaxed_hit_rank,
    }


def summarize_records(
    records: Sequence[dict],
    input_path: Path,
    token_threshold: float,
    fallback_top_k: int | None = None,
) -> dict:
    top_k = top_k_for_records(records, fallback=fallback_top_k)
    annotated = []
    for record in records:
        retrieval = record.get("retrieval") or {}
        target_rank = retrieval.get("target_title_rank")
        if not isinstance(target_rank, int):
            target_rank = None
        flags = evidence_flags(record, token_threshold)
        annotated.append(
            {
                "lang": record.get("lang") or "unknown",
                "target_title_rank": target_rank,
                "gold_page_hit": target_rank is not None,
                "reciprocal_rank": (1.0 / target_rank) if target_rank else 0.0,
                **flags,
            }
        )

    def summarize_subset(subset: Sequence[dict]) -> dict:
        count = len(subset)
        if not count:
            return {
                "records": 0,
                "gold_page_recall_at_k": 0.0,
                "mrr_at_k": 0.0,
                "strict_answer_string_recall_at_k": 0.0,
                "relaxed_answer_token_recall_at_k": 0.0,
            }
        gold_hits = sum(1 for item in subset if item["gold_page_hit"])
        strict_hits = sum(1 for item in subset if item["strict_answer_string_hit"])
        relaxed_hits = sum(1 for item in subset if item["relaxed_answer_token_hit"])
        return {
            "records": count,
            "gold_page_hits": gold_hits,
            "gold_page_recall_at_k": gold_hits / count,
            "mrr_at_k": sum(item["reciprocal_rank"] for item in subset) / count,
            "strict_answer_string_hits": strict_hits,
            "strict_answer_string_recall_at_k": strict_hits / count,
            "relaxed_answer_token_hits": relaxed_hits,
            "relaxed_answer_token_recall_at_k": relaxed_hits / count,
        }

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for item in annotated:
        by_lang[item["lang"]].append(item)

    return {
        "input_jsonl": str(input_path),
        "top_k": top_k,
        "record_count": len(records),
        "language_counts": language_counts(records),
        "token_overlap_threshold": token_threshold,
        "overall": summarize_subset(annotated),
        "per_language": {
            lang: summarize_subset(lang_items)
            for lang, lang_items in sorted(by_lang.items())
        },
    }


def default_input_paths(work_dir: Path) -> list[Path]:
    runs_dir = work_dir / "runs"
    return [runs_dir / name for name in DEFAULT_RETRIEVAL_FILES if (runs_dir / name).exists()]


def write_csv(summaries: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_jsonl",
        "top_k",
        "lang",
        "records",
        "gold_page_hits",
        "gold_page_recall_at_k",
        "mrr_at_k",
        "strict_answer_string_hits",
        "strict_answer_string_recall_at_k",
        "relaxed_answer_token_hits",
        "relaxed_answer_token_recall_at_k",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            rows = [("overall", summary["overall"])]
            rows.extend(summary["per_language"].items())
            for lang, metrics in rows:
                writer.writerow(
                    {
                        "input_jsonl": summary["input_jsonl"],
                        "top_k": summary["top_k"],
                        "lang": lang,
                        "records": metrics["records"],
                        "gold_page_hits": metrics.get("gold_page_hits", 0),
                        "gold_page_recall_at_k": f"{metrics['gold_page_recall_at_k']:.6f}",
                        "mrr_at_k": f"{metrics['mrr_at_k']:.6f}",
                        "strict_answer_string_hits": metrics.get(
                            "strict_answer_string_hits", 0
                        ),
                        "strict_answer_string_recall_at_k": (
                            f"{metrics['strict_answer_string_recall_at_k']:.6f}"
                        ),
                        "relaxed_answer_token_hits": metrics.get(
                            "relaxed_answer_token_hits", 0
                        ),
                        "relaxed_answer_token_recall_at_k": (
                            f"{metrics['relaxed_answer_token_recall_at_k']:.6f}"
                        ),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=PROJECT_WORK)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        action="append",
        help="Retrieval JSONL to evaluate. Defaults to top-5/top-10 dev retrieval files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_WORK
        / "summaries"
        / "e5_full_dev_retrieval_evidence_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_WORK
        / "summaries"
        / "e5_full_dev_retrieval_evidence_summary.csv",
    )
    parser.add_argument(
        "--token-overlap-threshold",
        type=float,
        default=0.7,
        help="Relaxed token hit threshold for answers with three or more unique tokens.",
    )
    parser.add_argument("--expected-count", type=int, default=1408)
    parser.add_argument("--allow-nonfull", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = args.input_jsonl or default_input_paths(args.work_dir)
    if not input_paths:
        raise FileNotFoundError(
            f"No retrieval files found. Pass --input-jsonl or check {args.work_dir / 'runs'}."
        )

    summaries = []
    for path in input_paths:
        records = read_jsonl(path)
        if args.expected_count and len(records) != args.expected_count and not args.allow_nonfull:
            raise ValueError(f"{path} has {len(records)} rows, expected {args.expected_count}")
        summary = summarize_records(records, path, args.token_overlap_threshold)
        summaries.append(summary)

    output = {
        "metric_notes": {
            "gold_page_recall_at_k": (
                "Fraction of questions where at least one retrieved chunk comes "
                "from the known development source page."
            ),
            "mrr_at_k": (
                "Mean reciprocal rank of the first retrieved chunk from the known "
                "development source page; misses receive 0."
            ),
            "strict_answer_string_recall_at_k": (
                "Lower-bound proxy: normalized full reference answer appears as a "
                "phrase in at least one retrieved chunk."
            ),
            "relaxed_answer_token_recall_at_k": (
                "Diagnostic proxy: enough normalized reference-answer tokens appear "
                "in the same retrieved chunk. This may still miss paraphrases and "
                "may overcount short/common answers."
            ),
        },
        "summaries": summaries,
    }
    write_json(output, args.output_json)
    write_csv(summaries, args.output_csv)
    print(f"Wrote retrieval evidence summary to {args.output_json}")
    print(f"Wrote retrieval evidence CSV to {args.output_csv}")
    for summary in summaries:
        overall = summary["overall"]
        print(
            f"k={summary['top_k']} records={summary['record_count']} "
            f"gold_recall={overall['gold_page_recall_at_k']:.4f} "
            f"mrr={overall['mrr_at_k']:.4f} "
            f"strict_answer={overall['strict_answer_string_recall_at_k']:.4f} "
            f"relaxed_answer={overall['relaxed_answer_token_recall_at_k']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
