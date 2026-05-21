#!/usr/bin/env python3
"""Build a Codabench submission zip from Embedding RAG predictions."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


REGION_TO_LANG = {
    "cs_CZ": "cs",
    "sk_SK": "sk",
    "uk_UA": "uk",
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prediction_key(record: dict) -> tuple[str, int]:
    lang = record.get("lang")
    record_id = record.get("id")
    if lang is None or record_id is None:
        raise ValueError(f"Prediction missing lang/id: {record}")
    return str(lang), int(record_id)


def load_predictions(predictions_jsonl: Path) -> dict[tuple[str, int], str]:
    predictions: dict[tuple[str, int], str] = {}
    for record in iter_jsonl(predictions_jsonl):
        key = prediction_key(record)
        answer = ((record.get("generation") or {}).get("answer") or "").strip()
        if key in predictions:
            raise ValueError(f"Duplicate prediction key: {key}")
        predictions[key] = answer
    return predictions


def build_submission_rows(
    template_jsonl: Path,
    predictions: dict[tuple[str, int], str],
    include_empty_template_rows: bool,
) -> list[dict]:
    rows = []
    used_keys = set()
    for template_record in iter_jsonl(template_jsonl):
        region = template_record.get("region")
        key = None
        if (
            template_record.get("split") == "test"
            and template_record.get("modality") == "text"
            and region in REGION_TO_LANG
        ):
            key = (REGION_TO_LANG[region], int(template_record["id"]))

        if key in predictions:
            answer = predictions[key]
            used_keys.add(key)
        elif include_empty_template_rows:
            answer = ""
        else:
            continue

        rows.append(
            {
                "id": template_record.get("id"),
                "split": "test",
                "modality": template_record.get("modality"),
                "region": region,
                "answer": answer,
            }
        )

    missing_template_keys = sorted(set(predictions) - used_keys)
    if missing_template_keys:
        raise ValueError(
            "Predictions not found in Codabench template: "
            f"{missing_template_keys[:20]}"
        )
    return rows


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_zip(jsonl_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, arcname=jsonl_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-jsonl",
        type=Path,
        default=Path("scripts/submission/cus_qa_test.jsonl"),
    )
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/runs/codabench_submission_text_native.jsonl"),
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("data/runs/codabench_submission_text_native.zip"),
    )
    parser.add_argument(
        "--include-empty-template-rows",
        action="store_true",
        help=(
            "Include all Codabench test rows, leaving unsupported modalities/"
            "regions empty. By default only rows with native-text predictions "
            "are written, which should be submitted with Codabench Partial=true."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_predictions(args.predictions_jsonl)
    rows = build_submission_rows(
        args.template_jsonl,
        predictions,
        args.include_empty_template_rows,
    )
    write_jsonl(rows, args.output_jsonl)
    write_zip(args.output_jsonl, args.output_zip)

    counts = Counter((row["modality"], row["region"]) for row in rows)
    print(f"Loaded {len(predictions)} predictions from {args.predictions_jsonl}")
    print(f"Wrote {len(rows)} Codabench rows to {args.output_jsonl}")
    print(f"Wrote zip to {args.output_zip}")
    print({str(key): counts[key] for key in sorted(counts)})
    if not args.include_empty_template_rows and len(rows) != 1399:
        raise ValueError(f"Expected 1399 native text rows, got {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
