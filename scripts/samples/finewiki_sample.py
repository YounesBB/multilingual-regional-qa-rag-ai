"""Sample FineWiki documents for smoke tests and schema checks."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

from datasets import load_dataset

DATASET_NAME = "HuggingFaceFW/finewiki"
CONFIG_BY_LANG = {"cs": "cs", "sk": "sk", "uk": "uk"}

FIELDS = (
    "id",
    "wikiname",
    "page_id",
    "title",
    "url",
    "date_modified",
    "in_language",
    "wikidata_id",
    "bytes_html",
    "version",
    "has_math",
    "text",
)


def resolve_configs(languages: Sequence[str]) -> list[tuple[str, str]]:
    pairs = []
    for lang in languages:
        config = CONFIG_BY_LANG.get(lang)
        if not config:
            raise ValueError(
                "Language must be one of: " + ", ".join(sorted(CONFIG_BY_LANG))
            )
        pairs.append((lang, config))
    return pairs


def iter_samples(
    lang: str,
    config: str,
    limit: int,
    cache_dir: Path | None,
    shuffle_buffer: int,
    seed: int,
):
    dataset = load_dataset(
        DATASET_NAME,
        name=config,
        split="train",
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return itertools.islice(dataset, limit)


def build_record(example: dict, lang: str) -> dict:
    record = {field: example.get(field) for field in FIELDS}
    record["lang"] = lang
    return record


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample FineWiki documents for smoke tests."
    )
    parser.add_argument(
        "--languages", nargs="+", default=["cs", "sk", "uk"], help="Languages to keep"
    )
    parser.add_argument(
        "--limit-per-lang",
        type=int,
        default=100,
        help="Number of documents per language",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Shuffle buffer size for streaming",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--output-jsonl", type=Path, required=True, help="Output JSONL path"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    args = parser.parse_args()

    if args.limit_per_lang <= 0:
        raise ValueError("limit per lang must be positive")

    records = []
    counts = {}
    for lang, config in resolve_configs(args.languages):
        samples = iter_samples(
            lang,
            config,
            args.limit_per_lang,
            args.cache_dir,
            args.shuffle_buffer,
            args.seed,
        )
        lang_records = [build_record(example, lang) for example in samples]
        records.extend(lang_records)
        counts[lang] = len(lang_records)

    write_jsonl(records, args.output_jsonl)
    summary = ", ".join(f"{lang}={counts[lang]}" for lang in args.languages)
    print(f"Wrote {len(records)} records to {args.output_jsonl} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
