"""Judge CUS-QA answers with Unbabel/M-Prometheus-7B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from full_dev_common import PROJECT_WORK, language_counts, limit_per_language, read_jsonl, record_identity, write_json
from smoke_rag import generate_answer, load_generator


RESULT_PATTERN = re.compile(
    r"\[RESULT\]\s*[:\-]?\s*(CORRECT|INCORRECT|[1-5])",
    flags=re.IGNORECASE,
)
LABEL_PATTERN = re.compile(r"\b(CORRECT|INCORRECT)\b", flags=re.IGNORECASE)
JUDGE_PROMPT_VERSION = "v2_semantic_harmless_extra"


def system_id(path: Path) -> str:
    return path.stem


def output_answer(record: dict) -> str:
    return str((record.get("generation") or {}).get("answer") or "").strip()


def reference_answer(record: dict) -> str:
    return str(record.get("reference_answer") or record.get("answer_orig") or "").strip()


def build_judge_prompt(record: dict) -> str:
    question = str(record.get("question") or record.get("question_orig") or "").strip()
    reference = reference_answer(record)
    answer = output_answer(record)
    return f"""###Task Description:
You are judging an open-ended factual question answering response.
Given a question, a reference answer, and a model answer, decide whether the model answer is semantically correct.
Accept equivalent wording, inflection, spelling variants, reordered lists, and harmless extra context if it does not contradict the reference.
If the model answer contains the required entity, place, person, date, number, or list from the reference, mark it CORRECT even if it also states the relation or activity type.
Do not penalize answers for adding relevant words such as "music", "theatre", "city", "district", or "metro" when the requested factual answer is still present.
Mark the answer INCORRECT only if it omits a required fact, adds a contradictory fact, answers a different question, or is too vague to verify.

First write brief feedback. Then write exactly one final label in this format:
[RESULT] CORRECT
or
[RESULT] INCORRECT

###Question:
{question}

###Reference Answer:
{reference}

###Model Answer:
{answer}

###Score Rubric:
CORRECT: The model answer contains the same factual answer as the reference, allowing paraphrases and harmless extra detail.
INCORRECT: The model answer is wrong, incomplete, unsupported, contradictory, or does not answer the question.

###Feedback:
"""


def parse_label(text: str) -> tuple[str | None, str | None]:
    result_match = RESULT_PATTERN.search(text)
    if result_match:
        value = result_match.group(1).upper()
        if value in {"4", "5"}:
            return "CORRECT", None
        if value in {"1", "2", "3"}:
            return "INCORRECT", None
        return value, None

    labels = [match.group(1).upper() for match in LABEL_PATTERN.finditer(text)]
    unique = set(labels)
    if len(unique) == 1:
        return labels[0], "label_without_result_marker"
    return None, "missing_or_ambiguous_label"


def existing_identities(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record_identity(record) for record in read_jsonl(path)}


def has_parsed_label(record: dict) -> bool:
    return (record.get("judge") or {}).get("label") in {"CORRECT", "INCORRECT"}


def rewrite_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def append_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def select_records(records: Sequence[dict], args: argparse.Namespace) -> list[dict]:
    records = limit_per_language(records, args.limit_records_per_lang)
    if args.limit_records > 0:
        records = list(records[: args.limit_records])
    return list(records)


def output_paths(input_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    sid = system_id(input_path)
    return (
        args.output_dir / f"mprometheus_{sid}.jsonl",
        args.output_dir / f"mprometheus_{sid}_summary.json",
    )


def judge_file(input_path: Path, generator, args: argparse.Namespace) -> dict:
    output_path, summary_path = output_paths(input_path, args)
    records = select_records(read_jsonl(input_path), args)
    if args.resume and args.retry_unparsed and output_path.exists():
        existing_records = read_jsonl(output_path)
        parsed_records = [record for record in existing_records if has_parsed_label(record)]
        if len(parsed_records) != len(existing_records):
            dropped = len(existing_records) - len(parsed_records)
            rewrite_jsonl(parsed_records, output_path)
            print(
                f"{system_id(input_path)}: dropped {dropped} unparsed existing judgments "
                f"before resume",
                flush=True,
            )
    completed = existing_identities(output_path) if args.resume else set()
    pending = [record for record in records if record_identity(record) not in completed]
    print(
        f"{system_id(input_path)}: records={len(records)}, "
        f"existing={len(completed)}, pending={len(pending)}",
        flush=True,
    )

    generated = 0
    for record in pending:
        prompt = build_judge_prompt(record)
        raw = generate_answer(
            generator.tokenizer,
            generator.model,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.generation_top_k,
            generator.uses_processor,
        )
        label, warning = parse_label(raw)
        judged = {
            "id": record.get("id"),
            "lang": record.get("lang"),
            "question": record.get("question") or record.get("question_orig"),
            "reference_answer": reference_answer(record),
            "model_answer": output_answer(record),
            "wikititle": record.get("wikititle"),
            "source_jsonl": str(input_path),
            "system_id": system_id(input_path),
            "judge": {
                "model_name": args.model_name,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "label": label,
                "is_correct": label == "CORRECT" if label else None,
                "parse_warning": warning,
                "raw_output": raw,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }
        append_jsonl([judged], output_path)
        generated += 1
        if generated == 1 or generated % args.log_every == 0:
            print(
                f"{system_id(input_path)}: judged {generated}/{len(pending)} pending",
                flush=True,
            )

    judged_records = read_jsonl(output_path)
    if args.limit_records_per_lang > 0 or args.limit_records > 0:
        selected_ids = {record_identity(record) for record in records}
        judged_records = [
            record for record in judged_records if record_identity(record) in selected_ids
        ]
    summary = summarize_judgments(judged_records, input_path, output_path, args)
    write_json(summary, summary_path)
    print(f"Wrote M-Prometheus summary to {summary_path}")
    return summary


def summarize_judgments(
    records: Sequence[dict],
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict:
    label_counts = Counter((record.get("judge") or {}).get("label") or "UNKNOWN" for record in records)
    correct = label_counts["CORRECT"]
    incorrect = label_counts["INCORRECT"]
    unknown = len(records) - correct - incorrect

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_lang[record.get("lang") or "unknown"].append(record)

    per_language = {}
    for lang, lang_records in sorted(by_lang.items()):
        lang_counts = Counter((record.get("judge") or {}).get("label") or "UNKNOWN" for record in lang_records)
        lang_total_known = lang_counts["CORRECT"] + lang_counts["INCORRECT"]
        per_language[lang] = {
            "record_count": len(lang_records),
            "correct": lang_counts["CORRECT"],
            "incorrect": lang_counts["INCORRECT"],
            "unknown": len(lang_records) - lang_total_known,
            "accuracy": lang_counts["CORRECT"] / lang_total_known
            if lang_total_known
            else 0.0,
        }

    total_known = correct + incorrect
    return {
        "system_id": system_id(input_path),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "judge_model_name": args.model_name,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "record_count": len(records),
        "language_counts": language_counts(records),
        "correct": correct,
        "incorrect": incorrect,
        "unknown": unknown,
        "accuracy": correct / total_known if total_known else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
        "per_language": per_language,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }


def write_combined_csv(summaries: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system_id",
        "record_count",
        "accuracy",
        "cs_accuracy",
        "sk_accuracy",
        "uk_accuracy",
        "correct",
        "incorrect",
        "unknown",
        "input_jsonl",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: item["accuracy"], reverse=True):
            per_lang = summary.get("per_language") or {}
            writer.writerow(
                {
                    "system_id": summary["system_id"],
                    "record_count": summary["record_count"],
                    "accuracy": f"{summary['accuracy']:.6f}",
                    "cs_accuracy": f"{per_lang.get('cs', {}).get('accuracy', 0.0):.6f}",
                    "sk_accuracy": f"{per_lang.get('sk', {}).get('accuracy', 0.0):.6f}",
                    "uk_accuracy": f"{per_lang.get('uk', {}).get('accuracy', 0.0):.6f}",
                    "correct": summary["correct"],
                    "incorrect": summary["incorrect"],
                    "unknown": summary["unknown"],
                    "input_jsonl": summary["input_jsonl"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_WORK / "eval" / "mprometheus_v2")
    parser.add_argument(
        "--combined-json",
        type=Path,
        default=PROJECT_WORK / "eval" / "mprometheus_v2_dev_summary.json",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=PROJECT_WORK / "eval" / "mprometheus_v2_dev_summary.csv",
    )
    parser.add_argument("--model-name", default="Unbabel/M-Prometheus-7B")
    parser.add_argument("--cache-dir", type=Path, default=Path("/fp/projects01/ec403/hf_models"))
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-records-per-lang", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--generation-top-k", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unparsed", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    start = time.time()
    generator = load_generator(
        args.model_name,
        args.cache_dir,
        args.trust_remote_code,
        args.local_files_only,
    )
    summaries = [judge_file(path, generator, args) for path in args.input_jsonl]
    combined = {
        "judge_model_name": args.model_name,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "elapsed_seconds": time.time() - start,
        "systems": summaries,
    }
    write_json(combined, args.combined_json)
    write_combined_csv(summaries, args.combined_csv)
    print(f"Wrote combined M-Prometheus JSON to {args.combined_json}")
    print(f"Wrote combined M-Prometheus CSV to {args.combined_csv}")

    unknown = sum(summary["unknown"] for summary in summaries)
    if unknown:
        raise ValueError(f"M-Prometheus produced {unknown} unparsed labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
