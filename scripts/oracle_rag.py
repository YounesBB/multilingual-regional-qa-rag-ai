"""Runs an Oracle RAG smoke test using exact Wikipedia source pages."""

from __future__ import annotations

import argparse
import hashlib
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
    load_generator,
    read_jsonl,
    retrieve,
    write_jsonl,
)
from wiki_text import rendered_html_to_text


API_USER_AGENT = "in5550-cusqa-rag-2026/1.0"


def cache_path(cache_dir: Path, lang: str, title: str) -> Path:
    normalized_title = title.replace(" ", "_")
    safe_title = quote(normalized_title, safe="")
    if len(safe_title) > 120:
        digest = hashlib.sha1(normalized_title.encode("utf-8")).hexdigest()[:16]
        safe_title = f"{safe_title[:80]}-{digest}"
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


def wikipedia_parse_url(lang: str, title: str) -> str:
    params = urlencode(
        {
            "action": "parse",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "text|displaytitle",
            "page": title,
        }
    )
    return f"https://{lang}.wikipedia.org/w/api.php?{params}"


def placeholder_page(
    lang: str,
    title: str,
    reason: str,
    page: Optional[dict] = None,
    rendered_fallback_attempted: bool = False,
) -> dict:
    page = page or {}
    return {
        "id": f"{lang}wiki/{page.get('pageid') or 'unknown'}",
        "page_id": page.get("pageid"),
        "title": page.get("title") or title.replace("_", " "),
        "url": page.get("fullurl")
        or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "lang": lang,
        "wikititle": title,
        "text": "",
        "source": "wikipedia_api_empty_or_missing",
        "fetch_warning": reason,
        "empty_extract": True,
        "rendered_fallback_attempted": rendered_fallback_attempted,
        "fetched_at_unix": int(time.time()),
    }


def fetch_rendered_wikipedia_page(
    lang: str,
    title: str,
    timeout: int,
    page: Optional[dict] = None,
) -> Optional[dict]:
    request = Request(
        wikipedia_parse_url(lang, title),
        headers={"User-Agent": API_USER_AGENT},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    parse = payload.get("parse") or {}
    rendered = parse.get("text") or ""
    if isinstance(rendered, dict):
        rendered = rendered.get("*") or ""
    text = rendered_html_to_text(rendered)
    if not text:
        return None

    page = page or {}
    resolved_title = parse.get("title") or page.get("title") or title
    page_id = parse.get("pageid") or page.get("pageid")
    return {
        "id": f"{lang}wiki/{page_id or 'unknown'}",
        "page_id": page_id,
        "title": resolved_title,
        "url": page.get("fullurl")
        or f"https://{lang}.wikipedia.org/wiki/{quote(resolved_title.replace(' ', '_'))}",
        "lang": lang,
        "wikititle": title,
        "text": text,
        "source": "wikipedia_api_parse_html",
        "fetch_warning": "empty_extract_used_rendered_fallback",
        "empty_extract": False,
        "rendered_fallback_attempted": True,
        "fetched_at_unix": int(time.time()),
    }


def fetch_wikipedia_page(lang: str, title: str, timeout: int) -> dict:
    request = Request(
        wikipedia_api_url(lang, title),
        headers={"User-Agent": API_USER_AGENT},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return placeholder_page(
            lang, title, "no_pages_returned", rendered_fallback_attempted=True
        )
    page = pages[0]
    if page.get("missing"):
        return placeholder_page(
            lang, title, "missing_page", page, rendered_fallback_attempted=True
        )

    text = page.get("extract") or ""
    if not text.strip():
        # Extracts can be empty for portal-like or heavily rendered pages.
        # Parse the rendered page before giving Oracle RAG an empty context.
        try:
            rendered_page = fetch_rendered_wikipedia_page(lang, title, timeout, page)
        except Exception as exc:
            rendered_page = None
            reason = f"empty_extract_rendered_fallback_failed:{type(exc).__name__}"
        else:
            reason = "empty_extract_rendered_fallback_empty"
        if rendered_page is not None:
            return rendered_page
        return placeholder_page(
            lang,
            title,
            reason,
            page,
            rendered_fallback_attempted=True,
        )

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
        "fetch_warning": None,
        "empty_extract": False,
        "rendered_fallback_attempted": False,
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
            cached_page = json.load(handle)
        old_empty_placeholder = (
            cached_page.get("empty_extract")
            and cached_page.get("source") == "wikipedia_api_empty_or_missing"
            and not cached_page.get("rendered_fallback_attempted")
        )
        # Empty cache entries written before the rendered fallback existed get
        # one fresh fetch so they do not keep hiding recoverable page text.
        if not old_empty_placeholder:
            return cached_page

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
        generator = load_generator(
            args.model_name,
            args.cache_dir,
            args.trust_remote_code,
            args.local_files_only,
        )

    output_records = []
    for question_record in questions:
        lang = question_record.get("lang") or ""
        question = question_record.get("question_orig") or question_record.get("question")
        page, retrieved = oracle_chunks_for_question(question_record, args)
        prompt = build_prompt(question, lang, retrieved)

        if generator:
            answer = generate_answer(
                generator.tokenizer,
                generator.model,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.generation_top_k,
                generator.uses_processor,
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
                    "empty_extract": page.get("empty_extract", False),
                    "fetch_warning": page.get("fetch_warning"),
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
                    "generation_top_k": args.generation_top_k,
                    "trust_remote_code": args.trust_remote_code,
                    "local_files_only": args.local_files_only,
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
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=0,
        help="Sampling top-k for generation. Use 0 to leave the model default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when required by a Hugging Face model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer files only from the local cache.",
    )
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
        os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir))
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
