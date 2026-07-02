"""Rerank existing FineWiki retrieval records with a cross-encoder model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from full_dev_common import (
    PROJECT_WORK,
    language_counts,
    limit_per_language,
    read_jsonl,
    write_json,
    write_jsonl,
)
from smoke_rag import build_prompt


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(unquote(title).replace("_", " ").casefold().split())


def default_output_path(input_jsonl: Path, reranker_tag: str, output_top_k: int) -> Path:
    stem = input_jsonl.stem
    return input_jsonl.with_name(f"{stem}_rerank_{reranker_tag}_top{output_top_k}.jsonl")


def default_summary_path(output_jsonl: Path) -> Path:
    work_runs = PROJECT_WORK / "runs"
    work_summaries = PROJECT_WORK / "summaries"
    try:
        relative = output_jsonl.relative_to(work_runs)
    except ValueError:
        return output_jsonl.with_name(f"{output_jsonl.stem}_summary.json")
    return work_summaries / f"{relative.stem}_summary.json"


def chunk_passage(chunk: dict) -> str:
    title = str(chunk.get("title") or "Untitled")
    text = str(chunk.get("text") or "")
    return f"Title: {title}\nText: {text}"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None,
        max_length: int,
        device: str | None,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Reranking requires torch and transformers. Load the Fox NLPL "
                "PyTorch and Transformers modules before running."
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def score(self, question: str, chunks: list[dict], batch_size: int) -> list[float]:
        scores: list[float] = []
        passages = [chunk_passage(chunk) for chunk in chunks]
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            batch_questions = [question] * len(batch_passages)
            encoded = self.tokenizer(
                batch_questions,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores


def rerank_record(record: dict, reranker: CrossEncoderReranker, args: argparse.Namespace) -> dict:
    retrieval = dict(record.get("retrieval") or {})
    chunks = list(retrieval.get("chunks") or [])
    if args.candidate_top_k > 0:
        chunks = chunks[: args.candidate_top_k]

    question = str(record.get("question") or record.get("question_orig") or "")
    scored_chunks = []
    scores = reranker.score(question, chunks, args.batch_size) if chunks else []
    for chunk, score in zip(chunks, scores):
        updated = dict(chunk)
        updated["original_rank"] = chunk.get("rank")
        updated["original_score"] = chunk.get("score")
        updated["reranker_score"] = score
        updated["score"] = score
        scored_chunks.append(updated)

    scored_chunks.sort(
        key=lambda chunk: (
            float(chunk.get("reranker_score") or 0.0),
            -int(chunk.get("original_rank") or 10**9),
        ),
        reverse=True,
    )

    selected = scored_chunks[: args.output_top_k]
    for rank, chunk in enumerate(selected, start=1):
        chunk["rank"] = rank

    target_title = normalize_title(record.get("wikititle"))
    target_title_rank = None
    for chunk in selected:
        if normalize_title(chunk.get("title")) == target_title:
            target_title_rank = chunk["rank"]
            break

    output_record = dict(record)
    retrieval.update(
        {
            "method": "e5_faiss_cross_encoder_rerank",
            "base_method": retrieval.get("method"),
            "reranker_model": args.reranker_model,
            "reranker_tag": args.reranker_tag,
            "candidate_top_k": len(chunks),
            "output_top_k": args.output_top_k,
            "top_k": args.output_top_k,
            "chunks": selected,
            "non_empty": bool(selected),
            "target_title_in_top_k": target_title_rank is not None,
            "target_title_rank": target_title_rank,
            "retrieved_titles": [normalize_title(chunk.get("title")) for chunk in selected],
        }
    )
    output_record["retrieval"] = retrieval
    output_record["prompt"] = build_prompt(question, output_record.get("lang") or "", selected)
    output_record["generation"] = {
        "skip_generation": True,
        "answer": "",
    }
    return output_record


def build_summary(records: list[dict], args: argparse.Namespace) -> dict:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_lang[record.get("lang") or "unknown"].append(record)

    language_summary = {}
    for lang, lang_records in sorted(by_lang.items()):
        title_hits = sum(
            1
            for record in lang_records
            if (record.get("retrieval") or {}).get("target_title_in_top_k")
        )
        target_ranks = [
            int((record.get("retrieval") or {}).get("target_title_rank"))
            for record in lang_records
            if (record.get("retrieval") or {}).get("target_title_rank") is not None
        ]
        language_summary[lang] = {
            "records": len(lang_records),
            "target_title_hits": title_hits,
            "target_title_hit_rate": title_hits / len(lang_records)
            if lang_records
            else 0.0,
            "target_title_mrr": sum(1.0 / rank for rank in target_ranks)
            / len(lang_records)
            if lang_records
            else 0.0,
        }

    target_hits = sum(
        1 for record in records if (record.get("retrieval") or {}).get("target_title_in_top_k")
    )
    target_ranks = [
        int((record.get("retrieval") or {}).get("target_title_rank"))
        for record in records
        if (record.get("retrieval") or {}).get("target_title_rank") is not None
    ]
    short_records = sum(
        1
        for record in records
        if len((record.get("retrieval") or {}).get("chunks") or []) < args.output_top_k
    )
    return {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "reranker_model": args.reranker_model,
        "reranker_tag": args.reranker_tag,
        "candidate_top_k": args.candidate_top_k,
        "output_top_k": args.output_top_k,
        "record_count": len(records),
        "language_counts": language_counts(records),
        "short_output_top_k_records": short_records,
        "target_title_hits": target_hits,
        "target_title_hit_rate": target_hits / len(records) if records else 0.0,
        "target_title_mrr": sum(1.0 / rank for rank in target_ranks) / len(records)
        if records
        else 0.0,
        "language_summary": language_summary,
        "tokenizer_max_length": args.tokenizer_max_length,
        "batch_size": args.batch_size,
        "limit_records": args.limit_records,
        "limit_records_per_lang": args.limit_records_per_lang,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_WORK / "runs" / "e5_full_dev_retrieval_top10.jsonl",
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-tag", default="bge")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_WORK / "hf_models")
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--output-top-k", type=int, default=5)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-records-per-lang", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--device")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output_top_k <= 0:
        parser.error("--output-top-k must be positive")
    if args.candidate_top_k < 0:
        parser.error("--candidate-top-k cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.output_jsonl is None:
        args.output_jsonl = default_output_path(
            args.input_jsonl,
            args.reranker_tag,
            args.output_top_k,
        )
    if args.summary_json is None:
        args.summary_json = default_summary_path(args.output_jsonl)
    return args


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    records = limit_per_language(records, args.limit_records_per_lang)
    if args.limit_records > 0:
        records = records[: args.limit_records]

    reranker = CrossEncoderReranker(
        args.reranker_model,
        args.cache_dir,
        args.tokenizer_max_length,
        args.device,
        args.trust_remote_code,
        args.local_files_only,
    )
    output_records = []
    for index, record in enumerate(records, start=1):
        output_records.append(rerank_record(record, reranker, args))
        if index == 1 or index % 100 == 0:
            print(f"Reranked {index}/{len(records)} records", flush=True)

    write_jsonl(output_records, args.output_jsonl)
    summary = build_summary(output_records, args)
    write_json(summary, args.summary_json)
    print(f"Wrote reranked retrieval to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
