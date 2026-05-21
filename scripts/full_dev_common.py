"""Shared helpers for full-development Oracle RAG runs."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_WORK = Path(f"/cluster/work/projects/ec403/{os.environ.get('USER', 'user')}/cusqa-rag-2026")
PROJECT_WORK = Path(os.environ.get("CUSQA_WORK", DEFAULT_WORK))
LANGUAGES = ("cs", "sk", "uk")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def language_counts(records: Sequence[dict]) -> dict[str, int]:
    counts = Counter(record.get("lang") or "unknown" for record in records)
    return {lang: counts[lang] for lang in sorted(counts)}


def limit_per_language(records: Sequence[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return list(records)
    counts: Counter[str] = Counter()
    selected = []
    for record in records:
        lang = record.get("lang") or ""
        if counts[lang] >= limit:
            continue
        selected.append(record)
        counts[lang] += 1
    return selected


def record_identity(record: dict) -> str:
    explicit_id = record.get("id")
    if explicit_id is not None:
        return f"{record.get('lang') or 'unknown'}::{explicit_id}"
    parts = [
        record.get("lang") or "",
        record.get("wikititle") or "",
        record.get("question") or record.get("question_orig") or "",
    ]
    return "::".join(parts)


def duplicate_identities(records: Sequence[dict]) -> list[str]:
    counts = Counter(record_identity(record) for record in records)
    return sorted(identity for identity, count in counts.items() if count > 1)


def required_path(record: dict, dotted_path: str):
    value = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def summarize_oracle_outputs(
    records: Sequence[dict],
    expected_records: Sequence[dict] | None = None,
    model_name: str | None = None,
) -> dict:
    required_paths = [
        "id",
        "lang",
        "question",
        "reference_answer",
        "wikititle",
        "oracle_page.title",
        "oracle_page.url",
        "retrieval.chunks",
        "prompt",
        "generation.model_name",
        "generation.answer",
    ]
    missing_required = Counter()
    empty_answer_ids = []
    for record in records:
        for path in required_paths:
            value = required_path(record, path)
            if value is None or value == "":
                missing_required[path] += 1
        answer = required_path(record, "generation.answer")
        if not isinstance(answer, str) or not answer.strip():
            empty_answer_ids.append(record_identity(record))

    duplicate_ids = duplicate_identities(records)
    summary = {
        "model_name": model_name,
        "record_count": len(records),
        "language_counts": language_counts(records),
        "empty_answer_count": len(empty_answer_ids),
        "empty_answer_ids": empty_answer_ids[:50],
        "duplicate_identity_count": len(duplicate_ids),
        "duplicate_identities": duplicate_ids[:50],
        "missing_required_counts": dict(sorted(missing_required.items())),
    }

    if expected_records is not None:
        expected_ids = {record_identity(record) for record in expected_records}
        output_ids = {record_identity(record) for record in records}
        summary.update(
            {
                "expected_record_count": len(expected_records),
                "expected_language_counts": language_counts(expected_records),
                "missing_output_count": len(expected_ids - output_ids),
                "extra_output_count": len(output_ids - expected_ids),
                "missing_output_ids": sorted(expected_ids - output_ids)[:50],
                "extra_output_ids": sorted(output_ids - expected_ids)[:50],
            }
        )

    return summary


def assert_valid_summary(summary: dict) -> None:
    errors = []
    expected = summary.get("expected_record_count")
    if expected is not None and summary["record_count"] != expected:
        errors.append(
            f"record_count={summary['record_count']} expected_record_count={expected}"
        )
    if summary["empty_answer_count"]:
        errors.append(f"empty_answer_count={summary['empty_answer_count']}")
    if summary["duplicate_identity_count"]:
        errors.append(f"duplicate_identity_count={summary['duplicate_identity_count']}")
    if summary.get("missing_output_count"):
        errors.append(f"missing_output_count={summary['missing_output_count']}")
    if summary.get("extra_output_count"):
        errors.append(f"extra_output_count={summary['extra_output_count']}")
    if any(summary["missing_required_counts"].values()):
        errors.append(f"missing_required_counts={summary['missing_required_counts']}")
    if errors:
        raise ValueError("; ".join(errors))
