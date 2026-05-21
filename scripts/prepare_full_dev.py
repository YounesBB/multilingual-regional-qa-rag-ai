"""Prepare full CUS-QA development input JSONL for Oracle RAG."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

from datasets import get_dataset_split_names, load_dataset

from full_dev_common import (
    LANGUAGES,
    PROJECT_WORK,
    language_counts,
    limit_per_language,
    write_jsonl,
)
from samples.cusqa_data import CONFIG_BY_LANG, DATASET_NAME, resolve_split_name


def add_full_dev_metadata(record: dict, lang: str, config: str, index: int) -> dict:
    item = dict(record)
    item["lang"] = lang
    item["config"] = config
    item["full_dev_index"] = index
    if item.get("id") is None:
        item["id"] = f"{lang}-dev-{index:06d}"

    wikititle = item.get("wikititle")
    if wikititle:
        item["wiki_url"] = f"https://{lang}.wikipedia.org/wiki/{quote(wikititle)}"
    else:
        item["wiki_url"] = None
    return item


def load_full_dev(
    split: str,
    languages: list[str],
    cache_dir: Path | None,
) -> list[dict]:
    records = []
    for lang in languages:
        config = CONFIG_BY_LANG[lang]
        available_splits = get_dataset_split_names(DATASET_NAME, config)
        split_name = resolve_split_name(split, available_splits)
        dataset = load_dataset(
            DATASET_NAME,
            config,
            split=split_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        base_index = len(records)
        records.extend(
            add_full_dev_metadata(record, lang, config, base_index + offset)
            for offset, record in enumerate(dataset)
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_WORK / "inputs" / "cusqa_dev_all.jsonl",
    )
    parser.add_argument("--limit-per-lang", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    unknown = sorted(set(args.languages) - set(CONFIG_BY_LANG))
    if unknown:
        raise ValueError(f"Unknown languages: {unknown}")

    records = load_full_dev(args.split, args.languages, args.cache_dir)
    records = limit_per_language(records, args.limit_per_lang)
    write_jsonl(records, args.output_jsonl)
    print(
        f"Wrote {len(records)} full-dev input records to {args.output_jsonl}: "
        f"{language_counts(records)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
