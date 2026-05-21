"""Compute chrF-style development scores for RAG output JSONL files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from full_dev_common import PROJECT_WORK, duplicate_identities, language_counts, read_jsonl, write_json


DEFAULT_SYSTEMS = [
    "oracle_full_dev_tiny_aya.jsonl",
    "oracle_full_dev_tiny_aya_water.jsonl",
    "oracle_full_dev_llama31_8b.jsonl",
    "oracle_full_dev_qwen3_30b_a3b.jsonl",
    "embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top5.jsonl",
    "embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top10.jsonl",
    "embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5.jsonl",
    "embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10.jsonl",
    "embedding_rag_llama31_8b_e5_full_dev_retrieval_top5.jsonl",
    "embedding_rag_llama31_8b_e5_full_dev_retrieval_top10.jsonl",
    "embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top5.jsonl",
    "embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10.jsonl",
]


def system_id(path: Path) -> str:
    return path.stem


def output_answer(record: dict) -> str:
    generation = record.get("generation") or {}
    return str(generation.get("answer") or "").strip()


def reference_answer(record: dict) -> str:
    return str(record.get("reference_answer") or record.get("answer_orig") or "").strip()


def normalize_text(text: str, keep_whitespace: bool) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if keep_whitespace:
        return text
    return re.sub(r"\s+", "", text)


def char_ngrams(text: str, order: int) -> Counter[str]:
    if order <= 0 or len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def corpus_chrf(
    hypotheses: Sequence[str],
    references: Sequence[str],
    char_order: int,
    beta: float,
    keep_whitespace: bool,
) -> dict:
    matches_by_order = [0] * char_order
    hyp_by_order = [0] * char_order
    ref_by_order = [0] * char_order

    for hypothesis, reference in zip(hypotheses, references):
        hyp_text = normalize_text(hypothesis, keep_whitespace)
        ref_text = normalize_text(reference, keep_whitespace)
        for order in range(1, char_order + 1):
            hyp_counts = char_ngrams(hyp_text, order)
            ref_counts = char_ngrams(ref_text, order)
            matches = sum((hyp_counts & ref_counts).values())
            matches_by_order[order - 1] += matches
            hyp_by_order[order - 1] += sum(hyp_counts.values())
            ref_by_order[order - 1] += sum(ref_counts.values())

    effective_orders = [
        index
        for index in range(char_order)
        if hyp_by_order[index] > 0 or ref_by_order[index] > 0
    ]
    if not effective_orders:
        return {
            "chrf": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "effective_char_order": 0,
        }

    precisions = [
        matches_by_order[index] / hyp_by_order[index]
        if hyp_by_order[index]
        else 0.0
        for index in effective_orders
    ]
    recalls = [
        matches_by_order[index] / ref_by_order[index]
        if ref_by_order[index]
        else 0.0
        for index in effective_orders
    ]
    precision = sum(precisions) / len(effective_orders)
    recall = sum(recalls) / len(effective_orders)
    beta2 = beta * beta
    if precision == 0.0 and recall == 0.0:
        score = 0.0
    else:
        score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    return {
        "chrf": score * 100.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "effective_char_order": max(index + 1 for index in effective_orders),
    }


def summarize_file(path: Path, args: argparse.Namespace) -> dict:
    records = read_jsonl(path)
    if args.expected_count and len(records) != args.expected_count and not args.allow_nonfull:
        raise ValueError(f"{path} has {len(records)} rows, expected {args.expected_count}")
    duplicates = duplicate_identities(records)
    references = [reference_answer(record) for record in records]
    hypotheses = [output_answer(record) for record in records]
    empty_answers = sum(1 for answer in hypotheses if not answer)
    missing_refs = sum(1 for reference in references if not reference)

    summary = {
        "system_id": system_id(path),
        "input_jsonl": str(path),
        "record_count": len(records),
        "language_counts": language_counts(records),
        "empty_answer_count": empty_answers,
        "missing_reference_count": missing_refs,
        "duplicate_identity_count": len(duplicates),
        "duplicate_identities": duplicates[:50],
    }
    summary.update(
        corpus_chrf(
            hypotheses,
            references,
            args.char_order,
            args.beta,
            args.keep_whitespace,
        )
    )

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_lang[record.get("lang") or "unknown"].append(record)

    per_language = {}
    for lang, lang_records in sorted(by_lang.items()):
        lang_hypotheses = [output_answer(record) for record in lang_records]
        lang_references = [reference_answer(record) for record in lang_records]
        lang_summary = {
            "record_count": len(lang_records),
            "empty_answer_count": sum(1 for answer in lang_hypotheses if not answer),
        }
        lang_summary.update(
            corpus_chrf(
                lang_hypotheses,
                lang_references,
                args.char_order,
                args.beta,
                args.keep_whitespace,
            )
        )
        per_language[lang] = lang_summary
    summary["per_language"] = per_language

    retrieval_hits = [
        bool((record.get("retrieval") or {}).get("target_title_in_top_k"))
        for record in records
        if "target_title_in_top_k" in (record.get("retrieval") or {})
    ]
    if retrieval_hits:
        summary["target_title_hit_rate"] = sum(retrieval_hits) / len(retrieval_hits)
        summary["target_title_hits"] = sum(retrieval_hits)

    return summary


def default_input_paths(runs_dir: Path) -> list[Path]:
    return [runs_dir / name for name in DEFAULT_SYSTEMS if (runs_dir / name).exists()]


def write_combined_csv(summaries: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system_id",
        "record_count",
        "chrf",
        "cs_chrf",
        "sk_chrf",
        "uk_chrf",
        "empty_answer_count",
        "target_title_hit_rate",
        "input_jsonl",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: item["chrf"], reverse=True):
            per_lang = summary.get("per_language") or {}
            writer.writerow(
                {
                    "system_id": summary["system_id"],
                    "record_count": summary["record_count"],
                    "chrf": f"{summary['chrf']:.4f}",
                    "cs_chrf": f"{per_lang.get('cs', {}).get('chrf', 0.0):.4f}",
                    "sk_chrf": f"{per_lang.get('sk', {}).get('chrf', 0.0):.4f}",
                    "uk_chrf": f"{per_lang.get('uk', {}).get('chrf', 0.0):.4f}",
                    "empty_answer_count": summary["empty_answer_count"],
                    "target_title_hit_rate": (
                        f"{summary['target_title_hit_rate']:.6f}"
                        if "target_title_hit_rate" in summary
                        else ""
                    ),
                    "input_jsonl": summary["input_jsonl"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_WORK / "runs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_WORK / "eval")
    parser.add_argument(
        "--combined-json",
        type=Path,
        default=PROJECT_WORK / "eval" / "chrf_dev_summary.json",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=PROJECT_WORK / "eval" / "chrf_dev_summary.csv",
    )
    parser.add_argument("--expected-count", type=int, default=1408)
    parser.add_argument("--allow-nonfull", action="store_true")
    parser.add_argument("--char-order", type=int, default=6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--keep-whitespace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = args.input_jsonl or default_input_paths(args.runs_dir)
    if not input_paths:
        raise ValueError("No input JSONL files found")

    summaries = []
    for path in input_paths:
        summary = summarize_file(path, args)
        summaries.append(summary)
        per_system_path = args.output_dir / f"{summary['system_id']}_chrf_summary.json"
        write_json(summary, per_system_path)
        print(
            f"{summary['system_id']}: chrF={summary['chrf']:.4f}, "
            f"records={summary['record_count']}, empty={summary['empty_answer_count']}"
        )

    combined = {
        "metric": "chrF",
        "char_order": args.char_order,
        "effective_order": True,
        "beta": args.beta,
        "keep_whitespace": args.keep_whitespace,
        "systems": sorted(summaries, key=lambda item: item["chrf"], reverse=True),
    }
    write_json(combined, args.combined_json)
    write_combined_csv(summaries, args.combined_csv)
    print(f"Wrote combined chrF JSON to {args.combined_json}")
    print(f"Wrote combined chrF CSV to {args.combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
