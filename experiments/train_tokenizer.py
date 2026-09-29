"""Train a byte-level BPE tokenizer on a bounded HF dataset stream."""

import argparse
from dataclasses import dataclass
from itertools import chain
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SIZE_UNITS = {"B": 1, "KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000}


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)\s*(B|KB|MB|GB)", value.upper())
    if match is None:
        raise argparse.ArgumentTypeError("size must look like 1MB, 10MB, 1GB, or 10GB")
    return int(match.group(1)) * SIZE_UNITS[match.group(2)]


@dataclass
class CorpusStats:
    bytes_used: int = 0
    documents_used: int = 0


def limited_text(
    rows: Iterable[dict], column: str, max_bytes: int, stats: CorpusStats
) -> Iterator[str]:
    for row in rows:
        if column not in row:
            raise ValueError(f"Text column {column!r} is missing; available: {', '.join(row)}")
        text = row[column]
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError(f"Text column {column!r} must contain strings, got {type(text).__name__}")
        if not text:
            continue

        encoded = text.encode("utf-8")
        remaining = max_bytes - stats.bytes_used
        truncated = len(encoded) > remaining
        if truncated:
            # Drop an incomplete final UTF-8 character at the byte boundary.
            text = encoded[:remaining].decode("utf-8", errors="ignore")
            encoded = text.encode("utf-8")
        if text:
            stats.bytes_used += len(encoded)
            stats.documents_used += 1
            yield text
        if stats.bytes_used >= max_bytes or truncated:
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--config", help="Dataset configuration, if needed")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument("--size", required=True, type=parse_size, help="UTF-8 corpus cap, e.g. 1MB or 10GB")
    parser.add_argument("--vocab-size", type=int, required=True, help="BPE vocabulary size")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for run artifacts")
    return parser


def train_tokenizer(
    dataset_id: str,
    config: str | None,
    split: str,
    text_column: str,
    size: int,
    vocab_size: int,
    output_dir: Path,
) -> dict:
    if vocab_size < 257:
        raise ValueError("vocab size must be at least 257 (256 bytes plus [UNK])")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {output_dir}")

    dataset = load_dataset(dataset_id, name=config, split=split, streaming=True)
    stats = CorpusStats()
    corpus = limited_text(dataset, text_column, size, stats)
    first = next(corpus, None)
    if first is None:
        raise ValueError("dataset yielded no nonempty text within the size limit")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(chain((first,), corpus), trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    summary = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "text_column": text_column,
        "byte_limit": size,
        "bytes_used": stats.bytes_used,
        "documents_used": stats.documents_used,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {output_dir / 'tokenizer.json'}", flush=True)
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        train_tokenizer(
            dataset_id=args.dataset,
            config=args.config,
            split=args.split,
            text_column=args.text_column,
            size=args.size,
            vocab_size=args.vocab_size,
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
