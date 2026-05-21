"""Validate full-development Oracle RAG outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from full_dev_common import (
    assert_valid_summary,
    read_jsonl,
    record_identity,
    summarize_oracle_outputs,
    write_json,
    write_jsonl,
)


def load_expected(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    return read_jsonl(path)


def validate_command(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input_jsonl)
    expected = load_expected(args.expected_jsonl)
    summary = summarize_oracle_outputs(records, expected, args.model_name)
    summary["input_jsonl"] = str(args.input_jsonl)
    if args.expected_jsonl:
        summary["expected_jsonl"] = str(args.expected_jsonl)
    write_json(summary, args.summary_json)
    print(f"Wrote validation summary to {args.summary_json}")
    print(summary)
    if not args.allow_invalid:
        assert_valid_summary(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--expected-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--allow-invalid", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return validate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
