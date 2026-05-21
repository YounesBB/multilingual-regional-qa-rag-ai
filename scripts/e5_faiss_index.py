"""Shared E5 embedding and FAISS indexing helpers for FineWiki retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from smoke_rag import Chunk


def import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not available in this Python environment. On Fox, create "
            "a project venv after loading the NLPL modules:\n"
            "  python -m venv --system-site-packages $CUSQA_WORK/venvs/faiss-cpu\n"
            "  source $CUSQA_WORK/venvs/faiss-cpu/bin/activate\n"
            "  python -m pip install --upgrade pip\n"
            "  python -m pip install faiss-cpu"
        ) from exc
    return faiss


def atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def atomic_write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


class AtomicJsonlWriter:
    """Write JSONL to a temp file and publish it only after a full success."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.handle = self.tmp_path.open("w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.handle.close()

    def publish(self) -> None:
        self.close()
        self.tmp_path.replace(self.path)

    def discard(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.tmp_path.unlink(missing_ok=True)


class E5Embedder:
    """Encode passages or queries with multilingual E5 using Transformers."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None,
        max_length: int,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "E5 embedding requires torch and transformers. Load the Fox "
                "NLPL PyTorch and Transformers modules before running."
            ) from exc

        self.torch = torch
        self.functional = functional
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {"local_files_only": local_files_only}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def encode_passages(self, texts: Sequence[str], batch_size: int) -> object:
        return self._encode([f"passage: {text}" for text in texts], batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int) -> object:
        return self._encode([f"query: {text}" for text in texts], batch_size)

    def _encode(self, texts: Sequence[str], batch_size: int) -> object:
        arrays = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            pooled = mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                self.torch,
            )
            # Normalized vectors make FAISS inner-product search equivalent to
            # cosine similarity for both passage and query embeddings.
            pooled = self.functional.normalize(pooled, p=2, dim=1)
            arrays.append(pooled.detach().cpu().float().numpy())
        if not arrays:
            raise ValueError("Cannot encode an empty text sequence")

        import numpy as np

        return np.concatenate(arrays, axis=0).astype("float32", copy=False)


def mean_pool(last_hidden_state, attention_mask, torch_module):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def create_inner_product_index(dimension: int):
    faiss = import_faiss()
    return faiss.IndexFlatIP(dimension)


def write_faiss_index(index, path: Path) -> None:
    faiss = import_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(path)


def chunk_to_metadata(chunk: Chunk) -> dict:
    return asdict(chunk)
