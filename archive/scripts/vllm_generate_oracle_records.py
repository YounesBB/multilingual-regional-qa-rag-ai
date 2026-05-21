"""Generate answers for Oracle RAG prompt records with vLLM.

The input JSONL should be produced by scripts/oracle_rag.py with
--skip-generation. This script preserves the Oracle output schema and fills the
generation.answer field using a vLLM offline batch engine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from full_dev_common import read_jsonl, record_identity


CHANNEL_TAG_PATTERN = re.compile(
    r"<\|channel\>thought\s*.*?<channel\|>",
    flags=re.DOTALL,
)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
HTMLISH_TAG_PATTERN = re.compile(r"<[^>]+>")


def append_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def clean_answer(text: str) -> str:
    text = CHANNEL_TAG_PATTERN.sub("", text)
    text = SPECIAL_TOKEN_PATTERN.sub("", text)
    text = HTMLISH_TAG_PATTERN.sub("", text)
    return text.strip()


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def pending_records(input_records: Sequence[dict], existing_records: Sequence[dict]):
    completed = {
        record_identity(record)
        for record in existing_records
    }
    return [
        record
        for record in input_records
        if record_identity(record) not in completed
    ]


def update_record(
    record: dict,
    answer: str,
    args: argparse.Namespace,
) -> dict:
    updated = dict(record)
    generation = dict(updated.get("generation") or {})
    generation.update(
        {
            "model_name": args.model_name,
            "backend": "vllm",
            "skip_generation": False,
            "fetch_only": False,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generation_top_k": args.top_k,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "vllm_max_model_len": args.max_model_len,
            "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_max_num_seqs": args.max_num_seqs,
            "vllm_max_num_batched_tokens": args.max_num_batched_tokens,
            "vllm_enforce_eager": args.enforce_eager,
            "vllm_version": args.vllm_version,
            "answer": answer,
        }
    )
    updated["generation"] = generation
    return updated


def batched(records: Sequence[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_messages(records: Sequence[dict]) -> list[list[dict]]:
    messages = []
    for record in records:
        prompt = record.get("prompt")
        if not prompt:
            identity = record_identity(record)
            raise ValueError(f"Input record is missing prompt: {identity}")
        messages.append([{"role": "user", "content": prompt}])
    return messages


def output_texts(outputs) -> list[str]:
    answers = []
    for output in outputs:
        if not output.outputs:
            answers.append("")
            continue
        answers.append(clean_answer(output.outputs[0].text or ""))
    return answers


def generate(args: argparse.Namespace) -> int:
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import vllm as vllm_module
    from vllm import LLM, SamplingParams

    args.vllm_version = getattr(vllm_module, "__version__", "unknown")

    input_records = read_jsonl(args.input_jsonl)
    existing_records = load_existing(args.output_jsonl)
    to_generate = pending_records(input_records, existing_records)
    completed_count = len(input_records) - len(to_generate)

    print(f"Input records: {len(input_records)}")
    print(f"Existing output records: {len(existing_records)}")
    print(f"Existing completed input records: {completed_count}")
    print(f"Pending records: {len(to_generate)}")
    if not to_generate:
        print(f"Nothing to generate; output already complete at {args.output_jsonl}")
        return 0

    llm_kwargs = {
        "model": args.model_name,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "task": "generate",
    }
    if args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.cache_dir:
        llm_kwargs["download_dir"] = str(args.cache_dir)
    if args.disable_multimodal_inputs:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    args.sampling_params = sampling_params

    generated_count = 0
    for batch in batched(to_generate, args.batch_size):
        outputs = llm.chat(
            build_messages(batch),
            sampling_params=sampling_params,
            use_tqdm=args.use_tqdm,
            chat_template_content_format="string",
            add_generation_prompt=True,
        )
        answers = output_texts(outputs)
        generated_records = [
            update_record(record, answer, args)
            for record, answer in zip(batch, answers)
        ]
        append_jsonl(generated_records, args.output_jsonl)
        generated_count += len(generated_records)
        print(f"Generated {generated_count}/{len(to_generate)} pending records")

    print(f"Wrote vLLM Oracle records to {args.output_jsonl}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-multimodal-inputs",
        action="store_true",
        help="Do not pass the text-only multimodal limit to vLLM.",
    )
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()
    args.disable_multimodal_inputs = not args.allow_multimodal_inputs
    args.use_tqdm = not args.no_tqdm
    return args


def main() -> int:
    return generate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
