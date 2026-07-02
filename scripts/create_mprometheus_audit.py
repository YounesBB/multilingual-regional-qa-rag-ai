"""Create and summarize a small manual audit sheet for M-Prometheus judgments.

The audit is intended as a qualitative sanity check, not a native-speaker human
evaluation.  The generated CSV samples judged examples across systems,
languages, and M-Prometheus labels.  After filling `manual_label`,
`manual_confidence`, and `manual_notes`, run the script again with
`--summarize-csv` to compute agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from full_dev_common import PROJECT_WORK, read_jsonl, record_identity, write_json


SYSTEM_OUTPUTS = {
    "embedding_qwen3_30b_top10": (
        "eval/mprometheus_v2_embedding_qwen3_30b_a3b_top10/"
        "mprometheus_embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10.jsonl"
    ),
    "embedding_llama31_8b_top10": (
        "eval/mprometheus_v2_embedding_llama31_8b_top10/"
        "mprometheus_embedding_rag_llama31_8b_e5_full_dev_retrieval_top10.jsonl"
    ),
    "embedding_tiny_aya_water_top5": (
        "eval/mprometheus_v2_embedding_tiny_aya_water_top5/"
        "mprometheus_embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5.jsonl"
    ),
    "oracle_qwen3_30b": (
        "eval/mprometheus_v2_oracle_qwen3_30b_a3b/"
        "mprometheus_oracle_full_dev_qwen3_30b_a3b.jsonl"
    ),
}

LANG_ORDER = ("cs", "sk", "uk")
LABEL_ORDER = ("CORRECT", "INCORRECT")
CSV_FIELDS = [
    "audit_id",
    "system_short",
    "system_id",
    "lang",
    "id",
    "question",
    "question_en",
    "reference_answer",
    "answer_en",
    "model_answer",
    "mprometheus_label",
    "mprometheus_feedback",
    "wikititle",
    "retrieved_titles",
    "context_snippet",
    "manual_label",
    "manual_confidence",
    "manual_notes",
]


def compact_text(text: object, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def judge_feedback(record: dict) -> str:
    raw = str((record.get("judge") or {}).get("raw_output") or "").strip()
    return compact_text(raw.split("[RESULT]")[0], 700)


def load_source_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record_identity(record): record for record in read_jsonl(path)}


def load_dev_translation_lookup(work_dir: Path) -> dict[tuple[str, object], dict]:
    path = work_dir / "inputs" / "cusqa_dev_all.jsonl"
    if not path.exists():
        return {}
    lookup = {}
    for record in read_jsonl(path):
        lookup[(record.get("lang"), record.get("id"))] = record
    return lookup


def source_record_for(judgment: dict, cache: dict[Path, dict[str, dict]]) -> dict:
    source_path_text = judgment.get("source_jsonl")
    if not source_path_text:
        return {}
    source_path = Path(source_path_text)
    if source_path not in cache:
        cache[source_path] = load_source_records(source_path)
    return cache[source_path].get(record_identity(judgment), {})


def retrieval_titles(source_record: dict, max_titles: int = 5) -> str:
    chunks = ((source_record.get("retrieval") or {}).get("chunks") or [])[:max_titles]
    titles = []
    for chunk in chunks:
        rank = chunk.get("rank")
        title = chunk.get("title") or "Untitled"
        score = chunk.get("score")
        if isinstance(score, (int, float)):
            titles.append(f"{rank}:{title} ({score:.3f})")
        else:
            titles.append(f"{rank}:{title}")
    return " | ".join(titles)


def context_snippet(source_record: dict, max_chars: int = 700) -> str:
    chunks = (source_record.get("retrieval") or {}).get("chunks") or []
    if not chunks:
        chunks = (source_record.get("oracle_page") or {}).get("chunks") or []
    snippets = []
    for chunk in chunks[:2]:
        title = chunk.get("title") or source_record.get("wikititle") or "Context"
        text = compact_text(chunk.get("text"), max_chars // 2)
        if text:
            snippets.append(f"{title}: {text}")
    return compact_text(" || ".join(snippets), max_chars)


def available_judgment_paths(work_dir: Path) -> dict[str, Path]:
    return {
        name: work_dir / relative_path
        for name, relative_path in SYSTEM_OUTPUTS.items()
        if (work_dir / relative_path).exists()
    }


def sample_records(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    paths = available_judgment_paths(args.work_dir)
    dev_lookup = load_dev_translation_lookup(args.work_dir)
    if args.system:
        requested = set(args.system)
        paths = {name: path for name, path in paths.items() if name in requested}
        missing = requested - set(paths)
        if missing:
            raise FileNotFoundError(f"Missing requested audit systems: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No M-Prometheus judgment files found under {args.work_dir}")

    source_cache: dict[Path, dict[str, dict]] = {}
    selected = []
    for system_short, path in sorted(paths.items()):
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for judgment in read_jsonl(path):
            lang = judgment.get("lang") or "unknown"
            label = (judgment.get("judge") or {}).get("label") or "UNKNOWN"
            if lang in LANG_ORDER and label in LABEL_ORDER:
                buckets[(lang, label)].append(judgment)

        for lang in LANG_ORDER:
            for label in LABEL_ORDER:
                candidates = list(buckets[(lang, label)])
                if len(candidates) < args.per_bucket:
                    raise ValueError(
                        f"{system_short} {lang} {label} has only {len(candidates)} "
                        f"records, need {args.per_bucket}"
                    )
                rng.shuffle(candidates)
                for judgment in candidates[: args.per_bucket]:
                    source = source_record_for(judgment, source_cache)
                    dev_record = dev_lookup.get((judgment.get("lang"), judgment.get("id")), {})
                    selected.append(
                        {
                            "system_short": system_short,
                            "system_id": judgment.get("system_id"),
                            "lang": lang,
                            "id": judgment.get("id"),
                            "question": judgment.get("question"),
                            "question_en": dev_record.get("question_en", ""),
                            "reference_answer": judgment.get("reference_answer"),
                            "answer_en": dev_record.get("answer_en", ""),
                            "model_answer": judgment.get("model_answer"),
                            "mprometheus_label": label,
                            "mprometheus_feedback": judge_feedback(judgment),
                            "wikititle": judgment.get("wikititle"),
                            "retrieved_titles": retrieval_titles(source),
                            "context_snippet": context_snippet(source),
                            "manual_label": "",
                            "manual_confidence": "",
                            "manual_notes": "",
                        }
                    )

    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["audit_id"] = f"A{index:03d}"
    return selected


def write_audit_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def normalize_manual_label(value: str) -> str:
    label = str(value or "").strip().casefold()
    if label in {"correct", "c", "yes", "true", "1"}:
        return "CORRECT"
    if label in {"incorrect", "wrong", "i", "no", "false", "0"}:
        return "INCORRECT"
    if label in {"uncertain", "unsure", "?", "unknown"}:
        return "UNCERTAIN"
    return ""


def summarize_audit(path: Path, output_json: Path | None = None) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    labeled_rows = []
    for row in rows:
        manual = normalize_manual_label(row.get("manual_label", ""))
        mprom = str(row.get("mprometheus_label") or "").strip().upper()
        row["_manual_normalized"] = manual
        row["_mprometheus_normalized"] = mprom
        if manual:
            labeled_rows.append(row)

    judged_rows = [row for row in labeled_rows if row["_manual_normalized"] in LABEL_ORDER]
    agreements = [
        row
        for row in judged_rows
        if row["_manual_normalized"] == row["_mprometheus_normalized"]
    ]
    disagreements = [
        row
        for row in judged_rows
        if row["_manual_normalized"] != row["_mprometheus_normalized"]
    ]

    by_lang: dict[str, Counter] = defaultdict(Counter)
    by_system: dict[str, Counter] = defaultdict(Counter)
    for row in judged_rows:
        agreed = row["_manual_normalized"] == row["_mprometheus_normalized"]
        by_lang[row.get("lang") or "unknown"]["total"] += 1
        by_lang[row.get("lang") or "unknown"]["agree" if agreed else "disagree"] += 1
        by_system[row.get("system_short") or "unknown"]["total"] += 1
        by_system[row.get("system_short") or "unknown"]["agree" if agreed else "disagree"] += 1

    summary = {
        "audit_csv": str(path),
        "rows": len(rows),
        "labeled_rows": len(labeled_rows),
        "confident_judged_rows": len(judged_rows),
        "uncertain_rows": sum(
            1 for row in labeled_rows if row["_manual_normalized"] == "UNCERTAIN"
        ),
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "agreement_rate": len(agreements) / len(judged_rows) if judged_rows else 0.0,
        "manual_label_counts": dict(Counter(row["_manual_normalized"] for row in labeled_rows)),
        "mprometheus_label_counts": dict(Counter(row["_mprometheus_normalized"] for row in rows)),
        "by_language": {
            lang: {
                "total": counts["total"],
                "agreements": counts["agree"],
                "disagreements": counts["disagree"],
                "agreement_rate": counts["agree"] / counts["total"]
                if counts["total"]
                else 0.0,
            }
            for lang, counts in sorted(by_lang.items())
        },
        "by_system": {
            system: {
                "total": counts["total"],
                "agreements": counts["agree"],
                "disagreements": counts["disagree"],
                "agreement_rate": counts["agree"] / counts["total"]
                if counts["total"]
                else 0.0,
            }
            for system, counts in sorted(by_system.items())
        },
        "disagreement_audit_ids": [row["audit_id"] for row in disagreements],
    }
    if output_json:
        write_json(summary, output_json)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=PROJECT_WORK)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_WORK / "eval" / "mprometheus_manual_audit_sample.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_WORK / "eval" / "mprometheus_manual_audit_summary.json",
    )
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--system",
        action="append",
        choices=sorted(SYSTEM_OUTPUTS),
        help="Restrict to one or more audit systems. Defaults to all four.",
    )
    parser.add_argument(
        "--summarize-csv",
        type=Path,
        help="Summarize an already manually labeled audit CSV instead of sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize_csv:
        summary = summarize_audit(args.summarize_csv, args.summary_json)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    rows = sample_records(args)
    write_audit_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} audit rows to {args.output_csv}")
    print("Fill manual_label with correct/incorrect/uncertain, then run:")
    print(
        f"python scripts/create_mprometheus_audit.py --summarize-csv {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
