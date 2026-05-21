#!/usr/bin/env python3
"""Prepare native text CUS-QA test rows for the Embedding RAG pipeline.

Codabench public data uses regions such as ``cs_CZ`` and includes text,
visual, native-language, and English-translated rows. Our RAG experiments are
for native text only, so this script converts those rows to the internal
``lang``-based JSONL format expected by ``retrieve_finewiki_dev.py``.
"""

from __future__ import annotations

import argparse
import json
import os
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


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert(template_jsonl: Path) -> list[dict]:
    output = []
    for record in iter_jsonl(template_jsonl):
        region = record.get("region")
        if record.get("split") != "test":
            continue
        if record.get("modality") != "text":
            continue
        if region not in REGION_TO_LANG:
            continue

        output.append(
            {
                "id": record.get("id"),
                "split": "test",
                "modality": "text",
                "region": region,
                "lang": REGION_TO_LANG[region],
                "question": record.get("question") or "",
                "reference_answer": None,
                "answer": "",
                "wikititle": None,
                "wiki_url": None,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-jsonl",
        type=Path,
        default=Path("scripts/submission/cus_qa_test.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            os.environ.get(
                "CUSQA_WORK",
                f"/cluster/work/projects/ec403/{os.environ.get('USER', 'user')}/cusqa-rag-2026",
            )
        )
        / "inputs"
        / "cusqa_test_text_native.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = convert(args.template_jsonl)
    write_jsonl(records, args.output_jsonl)
    counts = Counter(record["lang"] for record in records)
    print(f"Wrote {len(records)} native text test rows to {args.output_jsonl}")
    print({lang: counts[lang] for lang in sorted(counts)})
    if len(records) != 1399:
        raise ValueError(f"Expected 1399 native text test rows, got {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
