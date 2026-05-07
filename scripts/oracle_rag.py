"""Runs an Oracle RAG smoke test using exact Wikipedia source pages."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from smoke_rag import (
    build_chunk_vectors,
    build_chunks,
    build_prompt,
    compute_idf,
    generate_answer,
    limit_questions_per_language,
    load_qwen_generator,
    read_jsonl,
    retrieve,
    write_jsonl,
)


API_USER_AGENT = "in5550-cusqa-rag-2026/0.1 (younesb@uio.no)"


def cache_path(cache_dir: Path, lang: str, title: str) -> Path:
    safe_title = quote(title.replace(" ", "_"), safe="")
    return cache_dir / lang / f"{safe_title}.json"


def wikipedia_api_url(lang: str, title: str) -> str:
    params = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "extracts|info",
            "explaintext": "1",
            "inprop": "url",
            "titles": title,
        }
    )
    return f"https://{lang}.wikipedia.org/w/api.php?{params}"


def fetch_wikipedia_page(lang: str, title: str, timeout: int) -> dict:
    request = Request(
        wikipedia_api_url(lang, title),
        headers={"User-Agent": API_USER_AGENT},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError(f"Wikipedia API returned no pages for {lang}:{title}")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"Wikipedia page is missing for {lang}:{title}")

    text = page.get("extract") or ""
    if not text.strip():
        raise RuntimeError(f"Wikipedia page has empty extract for {lang}:{title}")

    return {
        "id": f"{lang}wiki/{page.get('pageid')}",
        "page_id": page.get("pageid"),
        "title": page.get("title") or title,
        "url": page.get("fullurl")
        or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "lang": lang,
        "wikititle": title,
        "text": text,
        "source": "wikipedia_api",
        "fetched_at_unix": int(time.time()),
    }


def load_or_fetch_page(
    lang: str,
    title: str,
    cache_dir: Path,
    refresh_cache: bool,
    timeout: int,
) -> dict:
    path = cache_path(cache_dir, lang, title)
    if path.exists() and not refresh_cache:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    page = fetch_wikipedia_page(lang, title, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(page, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return page


def oracle_chunks_for_question(
    question_record: dict,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    lang = question_record.get("lang") or ""
    title = question_record.get("wikititle")
    if not lang or not title:
        raise ValueError("Oracle examples must include lang and wikititle")

    page = load_or_fetch_page(
        lang,
        title,
        args.oracle_cache_dir,
        args.refresh_cache,
        args.fetch_timeout,
    )
    chunks = build_chunks([page], args.chunk_tokens, args.overlap_tokens)
    idf = compute_idf(chunks)
    chunk_vectors = build_chunk_vectors(chunks, idf)
    question = question_record.get("question_orig") or question_record.get("question")
    retrieved = retrieve(question, lang, chunks, chunk_vectors, idf, args.top_k)
    return page, retrieved


def build_records(args: argparse.Namespace) -> list[dict]:
    questions = limit_questions_per_language(
        read_jsonl(args.questions_jsonl), args.limit_questions_per_lang
    )

    generator = None
    if not args.skip_generation and not args.fetch_only:
        generator = load_qwen_generator(args.model_name, args.cache_dir)

    output_records = []
    for question_record in questions:
        lang = question_record.get("lang") or ""
        question = question_record.get("question_orig") or question_record.get("question")
        page, retrieved = oracle_chunks_for_question(question_record, args)
        prompt = build_prompt(question, lang, retrieved)

        if generator:
            answer = generate_answer(
                generator[0],
                generator[1],
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
        else:
            answer = ""

        output_records.append(
            {
                "id": question_record.get("id"),
                "lang": lang,
                "question": question,
                "reference_answer": question_record.get("answer_orig"),
                "wikititle": question_record.get("wikititle"),
                "wiki_url": question_record.get("wiki_url"),
                "oracle_page": {
                    "title": page.get("title"),
                    "url": page.get("url"),
                    "page_id": page.get("page_id"),
                    "cache_path": str(
                        cache_path(args.oracle_cache_dir, lang, question_record["wikititle"])
                    ),
                    "source": page.get("source"),
                },
                "retrieval": {
                    "method": "oracle_page_tfidf_chunk_rerank",
                    "top_k": args.top_k,
                    "chunks": retrieved,
                },
                "prompt": prompt,
                "generation": {
                    "model_name": args.model_name,
                    "skip_generation": args.skip_generation or args.fetch_only,
                    "fetch_only": args.fetch_only,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "answer": answer,
                },
            }
        )
    return output_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an Oracle RAG smoke pipeline.")
    parser.add_argument(
        "--questions-jsonl",
        type=Path,
        default=Path("data/smoke/cusqa_dev_10_per_lang.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/runs/oracle_rag_qwen3_0p6b.jsonl"),
    )
    parser.add_argument("--oracle-cache-dir", type=Path, default=Path("data/oracle_cache"))
    parser.add_argument("--limit-questions-per-lang", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--fetch-timeout", type=int, default=30)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refetch Wikipedia pages even when cached records exist.",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Fetch and chunk Oracle pages without loading the generator.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Validate Oracle retrieval and output shape without generation.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", str(args.cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(args.cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))

    records = build_records(args)
    write_jsonl(records, args.output_jsonl)
    language_counts = Counter(record["lang"] for record in records)
    summary = ", ".join(
        f"{lang}={language_counts[lang]}" for lang in sorted(language_counts)
    )
    print(f"Wrote {len(records)} Oracle RAG records to {args.output_jsonl}")
    print(f"Language counts: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
