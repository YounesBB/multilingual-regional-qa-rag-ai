"""Audits Oracle page coverage before scaling Oracle RAG generation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

from oracle_rag import cache_path, load_or_fetch_page, oracle_chunks_for_question
from smoke_rag import limit_questions_per_language, read_jsonl, write_jsonl


TRAILING_PUNCTUATION = " \t\r\n.。．!！?？:：;；,，"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "a",
    "ako",
    "and",
    "for",
    "in",
    "of",
    "the",
    "v",
    "ve",
    "vo",
    "z",
    "zo",
    "та",
    "у",
    "в",
}


def normalize_title(title: str) -> str:
    title = unquote(title or "")
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title)
    return title.strip().casefold()


def answer_variants(answer: str | None) -> list[str]:
    if not answer:
        return []
    variants = [answer.strip(), answer.strip().strip(TRAILING_PUNCTUATION)]
    unique = []
    for variant in variants:
        if variant and variant not in unique:
            unique.append(variant)
    return unique


def contains_answer(text: str, answer: str | None) -> tuple[bool, str | None]:
    folded_text = (text or "").casefold()
    for variant in answer_variants(answer):
        if variant.casefold() in folded_text:
            return True, variant
    return False, None


def normalize_evidence_text(text: str) -> str:
    text = (text or "").casefold()
    # Road labels can mix Latin M and Cyrillic М in references and page text.
    text = re.sub(r"(?<!\w)[mм](?=\s*\d)", "m", text)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_evidence_text(text))


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalized_tokens(text)
        if token not in STOPWORDS and token.strip(TRAILING_PUNCTUATION)
    ]


def token_matches(answer_token: str, text_token: str) -> bool:
    if answer_token == text_token:
        return True
    if answer_token.isdigit() or text_token.isdigit():
        return False
    min_prefix = 4 if len(answer_token) <= 4 else 5
    if len(answer_token) < min_prefix or len(text_token) < min_prefix:
        return False
    return answer_token[:min_prefix] == text_token[:min_prefix]


def token_overlap_match(text: str, answer: str | None) -> tuple[bool, str | None]:
    answer_tokens = content_tokens(answer or "")
    if not answer_tokens:
        return False, None
    text_tokens = content_tokens(text)
    matched = []
    for answer_token in answer_tokens:
        if any(token_matches(answer_token, text_token) for text_token in text_tokens):
            matched.append(answer_token)

    if len(answer_tokens) == 1:
        is_match = len(matched) == 1
    else:
        is_match = len(matched) / len(answer_tokens) >= 0.6
    if is_match:
        return True, " ".join(matched)
    return False, None


def analyze_answer_match(text: str, answer: str | None) -> dict:
    exact_match, exact_variant = contains_answer(text, answer)
    if exact_match:
        return {
            "exact": True,
            "relaxed": True,
            "match_type": "exact",
            "matched_variant": exact_variant,
        }

    normalized_text = normalize_evidence_text(text)
    for variant in answer_variants(answer):
        normalized_variant = normalize_evidence_text(variant)
        if normalized_variant and normalized_variant in normalized_text:
            return {
                "exact": False,
                "relaxed": True,
                "match_type": "normalized",
                "matched_variant": variant,
            }

    overlap_match, overlap_variant = token_overlap_match(text, answer)
    if overlap_match:
        return {
            "exact": False,
            "relaxed": True,
            "match_type": "token_overlap",
            "matched_variant": overlap_variant,
        }

    return {
        "exact": False,
        "relaxed": False,
        "match_type": "absent",
        "matched_variant": None,
    }


def expected_url(lang: str, wikititle: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{quote(wikititle)}"


def audit_record(question_record: dict, args: argparse.Namespace) -> dict:
    lang = question_record.get("lang") or ""
    wikititle = question_record.get("wikititle") or ""
    answer = question_record.get("answer_orig")
    question = question_record.get("question_orig") or question_record.get("question")

    base_record = {
        "id": question_record.get("id"),
        "lang": lang,
        "question": question,
        "reference_answer": answer,
        "wikititle": wikititle,
        "wiki_url": question_record.get("wiki_url"),
    }

    try:
        page = load_or_fetch_page(
            lang,
            wikititle,
            args.oracle_cache_dir,
            args.refresh_cache,
            args.fetch_timeout,
        )
        _, chunks = oracle_chunks_for_question(question_record, args)
    except Exception as exc:
        return {
            **base_record,
            "fetch": {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            "coverage": {
                "answer_in_full_page": False,
                "answer_in_top_chunks": False,
                "relaxed_answer_in_full_page": False,
                "relaxed_answer_in_top_chunks": False,
                "full_page_match_type": "absent",
                "top_chunks_match_type": "absent",
            },
        }

    chunk_text = "\n".join(chunk.get("text") or "" for chunk in chunks)
    full_page_match = analyze_answer_match(page.get("text") or "", answer)
    chunk_match = analyze_answer_match(chunk_text, answer)
    title_match = normalize_title(page.get("title") or "") == normalize_title(wikititle)
    expected = expected_url(lang, wikititle)
    actual_url = page.get("url") or ""

    return {
        **base_record,
        "fetch": {
            "ok": True,
            "cache_path": str(cache_path(args.oracle_cache_dir, lang, wikititle)),
            "source": page.get("source"),
        },
        "oracle_page": {
            "title": page.get("title"),
            "url": actual_url,
            "page_id": page.get("page_id"),
            "title_matches_wikititle": title_match,
            "expected_url": expected,
        },
        "coverage": {
            "answer_in_full_page": full_page_match["exact"],
            "full_page_matched_variant": full_page_match["matched_variant"],
            "answer_in_top_chunks": chunk_match["exact"],
            "top_chunks_matched_variant": chunk_match["matched_variant"],
            "relaxed_answer_in_full_page": full_page_match["relaxed"],
            "relaxed_answer_in_top_chunks": chunk_match["relaxed"],
            "full_page_match_type": full_page_match["match_type"],
            "top_chunks_match_type": chunk_match["match_type"],
            "top_k": args.top_k,
            "num_chunks": len(chunks),
        },
        "top_chunks": [
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "text_preview": (chunk.get("text") or "")[: args.preview_chars],
            }
            for chunk in chunks
        ],
    }


def print_summary(records: list[dict]) -> None:
    counts_by_lang: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        lang = record.get("lang") or "unknown"
        counts = counts_by_lang[lang]
        counts["total"] += 1
        if record.get("fetch", {}).get("ok"):
            counts["fetched"] += 1
        if record.get("oracle_page", {}).get("title_matches_wikititle"):
            counts["title_match"] += 1
        if record.get("coverage", {}).get("answer_in_full_page"):
            counts["answer_in_full_page"] += 1
        if record.get("coverage", {}).get("answer_in_top_chunks"):
            counts["answer_in_top_chunks"] += 1
        if record.get("coverage", {}).get("relaxed_answer_in_full_page"):
            counts["relaxed_answer_in_full_page"] += 1
        if record.get("coverage", {}).get("relaxed_answer_in_top_chunks"):
            counts["relaxed_answer_in_top_chunks"] += 1

    for lang in sorted(counts_by_lang):
        counts = counts_by_lang[lang]
        print(
            " ".join(
                [
                    f"{lang}:",
                    f"total={counts['total']}",
                    f"fetched={counts['fetched']}",
                    f"title_match={counts['title_match']}",
                    f"answer_in_full_page={counts['answer_in_full_page']}",
                    f"answer_in_top_chunks={counts['answer_in_top_chunks']}",
                    f"relaxed_answer_in_full_page={counts['relaxed_answer_in_full_page']}",
                    f"relaxed_answer_in_top_chunks={counts['relaxed_answer_in_top_chunks']}",
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Oracle page answer coverage.")
    parser.add_argument(
        "--questions-jsonl",
        type=Path,
        default=Path("data/smoke/cusqa_dev_10_per_lang.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/runs/oracle_audit_10_per_lang.jsonl"),
    )
    parser.add_argument("--oracle-cache-dir", type=Path, default=Path("data/oracle_cache"))
    parser.add_argument("--limit-questions-per-lang", type=int, default=10)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-timeout", type=int, default=30)
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refetch Wikipedia pages even when cached records exist.",
    )
    args = parser.parse_args()

    questions = limit_questions_per_language(
        read_jsonl(args.questions_jsonl), args.limit_questions_per_lang
    )
    records = [audit_record(question, args) for question in questions]
    write_jsonl(records, args.output_jsonl)
    print(f"Wrote {len(records)} Oracle audit records to {args.output_jsonl}")
    print_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
