"""Measure BPE merge-prefix compression on cached held-out FineWeb text."""

import json
from pathlib import Path
from time import perf_counter
from typing import Iterator

from datasets import load_dataset
from tokenizers import Tokenizer

from experiments.tokenizer_eval_fineweb import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_EVAL_BYTES,
    DEFAULT_SKIP_BYTES,
)
from experiments.train_tokenizer import CorpusStats, limited_text


BATCH_BYTES = 16_000_000
MAX_UTF8_BOUNDARY_SHORTFALL = 3


def prepare_heldout_text(
    output_dir: Path,
    dataset_id: str = DEFAULT_DATASET,
    config: str | None = DEFAULT_CONFIG,
    split: str = "train",
    text_column: str = "text",
    skip_bytes: int = DEFAULT_SKIP_BYTES,
    eval_bytes: int = DEFAULT_EVAL_BYTES,
) -> dict:
    text_path = output_dir / "heldout.jsonl"
    metadata_path = output_dir / "heldout_meta.json"
    temporary_path = output_dir / "heldout.jsonl.part"
    if text_path.exists() and metadata_path.exists():
        print(f"Using cached held-out text: {text_path}", flush=True)
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if text_path.exists() or metadata_path.exists() or temporary_path.exists():
        raise ValueError(f"incomplete held-out text cache in {output_dir}")
    if skip_bytes < 0 or eval_bytes < 1:
        raise ValueError("skip bytes must be nonnegative and eval bytes must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Opening streamed dataset: {dataset_id} "
        f"({config or 'default'}, {split}/{text_column})",
        flush=True,
    )
    dataset = load_dataset(dataset_id, name=config, split=split, streaming=True)
    skipped = CorpusStats()
    if skip_bytes:
        print(f"Skipping the first {skip_bytes:,} UTF-8 bytes", flush=True)
        for _ in limited_text(
            dataset, text_column, skip_bytes, skipped, progress_label="Skipped"
        ):
            pass
        if skipped.bytes_used < skip_bytes - MAX_UTF8_BOUNDARY_SHORTFALL:
            raise ValueError("dataset ended before the requested held-out slice")

    print(f"Caching the next {eval_bytes:,} UTF-8 bytes", flush=True)
    cached = CorpusStats()
    with temporary_path.open("w", encoding="utf-8") as handle:
        for text in limited_text(
            dataset, text_column, eval_bytes, cached, progress_label="Cached"
        ):
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    if cached.bytes_used < eval_bytes - MAX_UTF8_BOUNDARY_SHORTFALL:
        raise ValueError("dataset ended before the requested evaluation size")
    temporary_path.replace(text_path)

    metadata = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "text_column": text_column,
        "skip_bytes_requested": skip_bytes,
        "skipped_utf8_bytes": skipped.bytes_used,
        "skipped_documents": skipped.documents_used,
        "eval_bytes_requested": eval_bytes,
        "cached_utf8_bytes": cached.bytes_used,
        "cached_documents": cached.documents_used,
        "text_path": str(text_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved held-out text to {text_path}", flush=True)
    return metadata


def load_prefix_tokenizer(source_path: Path, vocab_size: int) -> Tokenizer:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    model = source.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"source tokenizer is not BPE: {source_path}")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError(f"source tokenizer has an invalid BPE model: {source_path}")

    full_vocab_size = len(vocab)
    base_vocab_size = full_vocab_size - len(merges)
    if not 0 < base_vocab_size <= vocab_size <= full_vocab_size:
        raise ValueError(
            f"vocab size must be between {base_vocab_size:,} and {full_vocab_size:,}"
        )
    ids = set(vocab.values())
    if ids != set(range(full_vocab_size)):
        raise ValueError("BPE vocabulary IDs must be contiguous from zero")

    merge_count = vocab_size - base_vocab_size
    source["model"]["vocab"] = {
        token: token_id for token, token_id in vocab.items() if token_id < vocab_size
    }
    source["model"]["merges"] = merges[:merge_count]
    tokenizer = Tokenizer.from_str(json.dumps(source))
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"failed to construct the {vocab_size:,}-entry BPE prefix")
    return tokenizer


def vocabulary_sizes(source_path: Path, step: int) -> list[int]:
    if step < 1:
        raise ValueError("vocabulary step must be positive")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    model = source.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError(f"source tokenizer has an invalid BPE model: {source_path}")
    base_vocab_size = len(vocab) - len(merges)
    full_vocab_size = len(vocab)
    sizes = list(range(max(step, base_vocab_size), full_vocab_size + 1, step))
    if sizes[-1] != full_vocab_size:
        sizes.append(full_vocab_size)
    return sizes


def iter_cached_text(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line)
            if not isinstance(text, str):
                raise ValueError(f"cached text must contain JSON strings: {path}")
            yield text


def count_tokens(tokenizer: Tokenizer, text_path: Path) -> tuple[int, int, int]:
    token_count = 0
    text_bytes = 0
    document_count = 0
    batch: list[str] = []
    batch_bytes = 0

    def score_batch() -> int:
        return sum(len(encoding.ids) for encoding in tokenizer.encode_batch(batch))

    for text in iter_cached_text(text_path):
        encoded_size = len(text.encode("utf-8"))
        batch.append(text)
        batch_bytes += encoded_size
        text_bytes += encoded_size
        document_count += 1
        if batch_bytes >= BATCH_BYTES:
            token_count += score_batch()
            batch.clear()
            batch_bytes = 0
    if batch:
        token_count += score_batch()
    return token_count, text_bytes, document_count


def measure_compression(
    source_path: Path,
    heldout_dir: Path,
    output_path: Path,
    vocab_step: int = 16_000,
) -> dict:
    text_path = heldout_dir / "heldout.jsonl"
    metadata_path = heldout_dir / "heldout_meta.json"
    if not text_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"held-out text cache is missing from {heldout_dir}")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    if not source_path.is_file():
        raise ValueError(f"source tokenizer was not found: {source_path}")

    cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sizes = vocabulary_sizes(source_path, vocab_step)
    print(
        f"Measuring {len(sizes)} BPE prefixes from {sizes[0]:,} to {sizes[-1]:,} entries",
        flush=True,
    )
    started = perf_counter()
    points = []
    for vocab_size in sizes:
        tokenizer = load_prefix_tokenizer(source_path, vocab_size)
        point_started = perf_counter()
        token_count, text_bytes, document_count = count_tokens(tokenizer, text_path)
        elapsed = perf_counter() - point_started
        if token_count == 0:
            raise ValueError(f"the {vocab_size:,}-entry BPE prefix produced no tokens")
        point = {
            "vocab_size": vocab_size,
            "token_count": token_count,
            "bytes_per_token": text_bytes / token_count,
            "tokens_per_byte": token_count / text_bytes,
            "seconds": elapsed,
        }
        points.append(point)
        print(
            f"Vocab {vocab_size:,}: {point['bytes_per_token']:.4f} bytes/token "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    summary = {
        "source_tokenizer": str(source_path),
        "heldout_cache": cache_metadata,
        "vocab_step": vocab_step,
        "evaluated_utf8_bytes": text_bytes,
        "evaluated_documents": document_count,
        "total_seconds": perf_counter() - started,
        "points": points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved compression measurements to {output_path}", flush=True)
    return summary
