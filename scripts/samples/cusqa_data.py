"""Load CUS-QA splits and generate small samples for smoke tests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from datasets import concatenate_datasets, get_dataset_split_names, load_dataset

DATASET_NAME = "ufal/cus-qa"
CONFIG_BY_LANG = {"cs": "text-CZ", "sk": "text-SK", "uk": "text-UA"}
LANG_BY_CONFIG = {value: key for key, value in CONFIG_BY_LANG.items()}


def resolve_split_name(requested: str, available: Sequence[str]) -> str:
    if requested in available:
        return requested
    aliases = {
        "dev": ["validation", "valid"],
        "validation": ["dev", "valid"],
        "test": ["test"],
        "train": ["train"],
    }
    for alt in aliases.get(requested, []):
        if alt in available:
            return alt
    raise ValueError(
        f"Split '{requested}' is not available. Available splits: {sorted(available)}"
    )


def resolve_config_pairs(languages: Sequence[str], configs: Sequence[str] | None):
    if configs:
        pairs: list[tuple[str, str]] = []
        for config in configs:
            lang = LANG_BY_CONFIG.get(config)
            if not lang:
                raise ValueError(
                    "Config must be one of: " + ", ".join(sorted(LANG_BY_CONFIG))
                )
            pairs.append((lang, config))
        return pairs
    pairs = []
    for lang in languages:
        config = CONFIG_BY_LANG.get(lang)
        if not config:
            raise ValueError(
                "Language must be one of: " + ", ".join(sorted(CONFIG_BY_LANG))
            )
        pairs.append((lang, config))
    return pairs


def add_metadata(dataset, lang: str, config: str, add_wiki_url: bool):
    def _add_fields(example):
        example["lang"] = lang
        example["config"] = config
        if add_wiki_url:
            wikititle = example.get("wikititle")
            if wikititle:
                example["wiki_url"] = f"https://{lang}.wikipedia.org/wiki/{wikititle}"
            else:
                example["wiki_url"] = None
        return example

    return dataset.map(_add_fields)


def filter_languages(dataset, languages: Sequence[str]):
    language_set = set(languages)
    return dataset.filter(lambda ex: ex.get("lang") in language_set)


def sample_per_language(dataset, languages: Sequence[str], limit: int, seed: int):
    if limit <= 0:
        return dataset
    indices_by_lang: dict[str, list[int]] = {lang: [] for lang in languages}
    for idx, ex in enumerate(dataset):
        lang = ex.get("lang")
        if lang in indices_by_lang:
            indices_by_lang[lang].append(idx)
    rng = random.Random(seed)
    selected: list[int] = []
    for lang, indices in indices_by_lang.items():
        if not indices:
            continue
        if len(indices) <= limit:
            selected.extend(indices)
            continue
        selected.extend(rng.sample(indices, limit))
    return dataset.select(sorted(selected))


def count_by_language(dataset, languages: Sequence[str]) -> dict[str, int]:
    counts = {lang: 0 for lang in languages}
    for ex in dataset:
        lang = ex.get("lang")
        if lang in counts:
            counts[lang] += 1
    return counts


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load CUS QA splits and optionally write a language sample."
    )
    parser.add_argument("--split", default="dev", help="Split name to load")
    parser.add_argument(
        "--languages", nargs="+", default=["cs", "sk", "uk"], help="Languages to keep"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional CUS QA configs such as text-CZ text-SK text-UA",
    )
    parser.add_argument(
        "--limit-per-lang",
        type=int,
        default=0,
        help="Sample size per language. Zero keeps all",
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="Random seed for sampling"
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path for saving filtered records",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--add-wiki-url",
        action="store_true",
        help="Add wiki_url derived from wikititle when available",
    )
    args = parser.parse_args()

    config_pairs = resolve_config_pairs(args.languages, args.configs)
    active_languages = [lang for lang, _ in config_pairs]

    split_name = None
    datasets = []
    for lang, config in config_pairs:
        available_splits = get_dataset_split_names(DATASET_NAME, config)
        config_split = resolve_split_name(args.split, available_splits)
        if split_name is None:
            split_name = config_split
        elif split_name != config_split:
            raise ValueError("Split mismatch across configs")

        ds = load_dataset(
            DATASET_NAME,
            config,
            split=config_split,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        ds = add_metadata(ds, lang, config, args.add_wiki_url)
        datasets.append(ds)

    dataset = concatenate_datasets(datasets)
    dataset = filter_languages(dataset, active_languages)
    dataset = sample_per_language(
        dataset, active_languages, args.limit_per_lang, args.seed
    )

    counts = count_by_language(dataset, active_languages)
    total = sum(counts.values())
    summary = ", ".join(f"{lang}={counts[lang]}" for lang in active_languages)
    print(f"Loaded {total} records from split '{split_name}': {summary}")

    if args.output_jsonl:
        write_jsonl(dataset, args.output_jsonl)
        print(f"Wrote {total} records to {args.output_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
