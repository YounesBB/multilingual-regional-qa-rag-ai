"""Runs a small RAG smoke test over sampled CUS-QA and FineWiki data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


LANGUAGE_NAMES = {"cs": "Czech", "sk": "Slovak", "uk": "Ukrainian"}
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: Optional[str]
    url: Optional[str]
    lang: str
    text: str


@dataclass(frozen=True)
class Generator:
    tokenizer: object
    model: object
    uses_processor: bool = False


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def chunk_document(
    record: dict, chunk_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    text = normalize_whitespace(record.get("text") or "")
    tokens = text.split()
    if not tokens:
        return []
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    stride = chunk_tokens - overlap_tokens
    chunks = []
    doc_id = str(record.get("id") or record.get("page_id") or "unknown")
    for chunk_index, start in enumerate(range(0, len(tokens), stride)):
        piece = tokens[start : start + chunk_tokens]
        if not piece:
            continue
        chunk_text = " ".join(piece)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{chunk_index}",
                doc_id=doc_id,
                title=record.get("title"),
                url=record.get("url"),
                lang=record.get("lang") or record.get("in_language") or "",
                text=chunk_text,
            )
        )
        if start + chunk_tokens >= len(tokens):
            break
    return chunks


def build_chunks(
    corpus_records: Sequence[dict], chunk_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    chunks = []
    for record in corpus_records:
        chunks.extend(chunk_document(record, chunk_tokens, overlap_tokens))
    return chunks


def compute_idf(chunks: Sequence[Chunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(tokenize(chunk.text)))
    total = len(chunks)
    return {
        term: math.log((1 + total) / (1 + count)) + 1.0
        for term, count in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    if not counts:
        return {}
    length = sum(counts.values())
    return {
        term: (count / length) * idf.get(term, 1.0)
        for term, count in counts.items()
    }


def cosine_score(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_chunk_vectors(chunks: Sequence[Chunk], idf: dict[str, float]):
    return [vectorize(chunk.text, idf) for chunk in chunks]


def retrieve(
    question: str,
    lang: str,
    chunks: Sequence[Chunk],
    chunk_vectors: Sequence[dict[str, float]],
    idf: dict[str, float],
    top_k: int,
) -> list[dict]:
    query_vector = vectorize(question, idf)
    scored = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        if chunk.lang != lang:
            continue
        score = cosine_score(query_vector, chunk_vector)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "url": chunk.url,
                "lang": chunk.lang,
                "text": chunk.text,
            }
        )
    return results


def limit_questions_per_language(
    records: Sequence[dict], limit_per_lang: int
) -> list[dict]:
    if limit_per_lang <= 0:
        return list(records)
    counts: defaultdict[str, int] = defaultdict(int)
    selected = []
    for record in records:
        lang = record.get("lang") or ""
        if counts[lang] >= limit_per_lang:
            continue
        selected.append(record)
        counts[lang] += 1
    return selected


def build_prompt(question: str, lang: str, contexts: Sequence[dict]) -> str:
    language_name = LANGUAGE_NAMES.get(lang, lang)
    context_blocks = []
    for context in contexts:
        title = context.get("title") or "Untitled"
        text = context.get("text") or ""
        context_blocks.append(f"[{context['rank']}] {title}\n{text}")
    joined_context = "\n\n".join(context_blocks)
    return (
        f"Answer the question in {language_name}. Use only the context below. "
        "If the context is insufficient, give the best short answer supported by it.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{joined_context}\n\n"
        "Short answer:"
    )


def load_generator(
    model_name: str,
    cache_dir: Optional[Path],
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Generation requires torch and transformers. Use --skip-generation "
            "for local retrieval-only smoke checks."
        ) from exc

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
        tokenizer_kwargs["cache_dir"] = str(cache_dir)

    uses_processor = "gemma-4" in model_name.lower()
    if uses_processor:
        try:
            tokenizer = AutoProcessor.from_pretrained(model_name, **tokenizer_kwargs)
        except ValueError:
            uses_processor = False
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return Generator(tokenizer=tokenizer, model=model, uses_processor=uses_processor)


def load_qwen_generator(model_name: str, cache_dir: Optional[Path]):
    return load_generator(model_name, cache_dir)


def render_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return prompt


def _inputs_to_device(model_inputs, device):
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    return {key: value.to(device) for key, value in model_inputs.items()}


def _decode_generation(tokenizer, output_ids, uses_processor: bool) -> str:
    response = tokenizer.decode(output_ids, skip_special_tokens=not uses_processor)
    if uses_processor and hasattr(tokenizer, "parse_response"):
        try:
            parsed = tokenizer.parse_response(response)
        except Exception:
            parsed = None
        if isinstance(parsed, str):
            response = parsed
        elif isinstance(parsed, dict):
            response = (
                parsed.get("answer")
                or parsed.get("content")
                or parsed.get("text")
                or parsed.get("response")
                or response
            )
        elif parsed:
            response = str(parsed)

    response = re.sub(
        r"<\|channel\>thought\s*.*?<channel\|>",
        "",
        response,
        flags=re.DOTALL,
    )
    response = re.sub(r"<\|[^>]+?\|>", "", response)
    response = re.sub(r"<[^>]+>", "", response)
    return response.strip()


def generate_answer(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_top_k: int = 0,
    uses_processor: bool = False,
) -> str:
    text = render_chat_prompt(tokenizer, prompt)
    input_device = getattr(model, "device", None)
    if input_device is None:
        input_device = next(model.parameters()).device
    if uses_processor:
        model_inputs = tokenizer(text=text, return_tensors="pt")
    else:
        model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = _inputs_to_device(model_inputs, input_device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        if generation_top_k > 0:
            generation_kwargs["top_k"] = generation_top_k
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    generated_ids = model.generate(
        **model_inputs,
        **generation_kwargs,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return _decode_generation(tokenizer, output_ids, uses_processor)


def build_records(args: argparse.Namespace) -> list[dict]:
    questions = limit_questions_per_language(
        read_jsonl(args.questions_jsonl), args.limit_questions_per_lang
    )
    corpus_records = read_jsonl(args.corpus_jsonl)
    chunks = build_chunks(corpus_records, args.chunk_tokens, args.overlap_tokens)
    idf = compute_idf(chunks)
    chunk_vectors = build_chunk_vectors(chunks, idf)

    generator = None
    if not args.skip_generation:
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
        retrieved = retrieve(
            question, lang, chunks, chunk_vectors, idf, args.top_k
        )
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
                "retrieval": {
                    "method": "tfidf_cosine",
                    "top_k": args.top_k,
                    "chunks": retrieved,
                },
                "prompt": prompt,
                "generation": {
                    "model_name": args.model_name,
                    "skip_generation": args.skip_generation,
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
    parser = argparse.ArgumentParser(description="Run a smoke RAG pipeline.")
    parser.add_argument(
        "--questions-jsonl",
        type=Path,
        default=Path("data/smoke/cusqa_dev_10_per_lang.jsonl"),
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=Path("data/smoke/finewiki_sample_100_per_lang.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/runs/smoke_rag_qwen3_0p6b.jsonl"),
    )
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
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Validate retrieval and output shape without loading a generator.",
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
    print(f"Wrote {len(records)} smoke RAG records to {args.output_jsonl}")
    print(f"Language counts: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
