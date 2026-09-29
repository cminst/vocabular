"""Evaluate tokenizer compression statistics on a held-out FineWeb stream."""

import argparse
import json
from math import ceil, log2
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

from datasets import load_dataset
from tokenizers import Tokenizer

from experiments.train_tokenizer import CorpusStats, limited_text, parse_size


DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_CONFIG = "sample-10BT"
DEFAULT_SKIP_BYTES = 10_000_000_000
DEFAULT_EVAL_BYTES = 1_000_000_000
BATCH_BYTES = 16_000_000
MAX_UTF8_BOUNDARY_SHORTFALL = 3


def parse_tokenizer_spec(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokenizer must be NAME=PATH") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("tokenizer must be NAME=PATH")
    return name, Path(path)


def evaluate_tokenizers(
    tokenizer_paths: Mapping[str, Path],
    output_dir: Path,
    dataset_id: str = DEFAULT_DATASET,
    config: str | None = DEFAULT_CONFIG,
    split: str = "train",
    text_column: str = "text",
    skip_bytes: int = DEFAULT_SKIP_BYTES,
    eval_bytes: int = DEFAULT_EVAL_BYTES,
) -> dict:
    if not tokenizer_paths:
        raise ValueError("select at least one tokenizer")
    if skip_bytes < 0:
        raise ValueError("skip bytes must not be negative")
    if eval_bytes < 1:
        raise ValueError("eval bytes must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    tokenizers = {}
    for name, path in tokenizer_paths.items():
        if not path.is_file():
            raise ValueError(f"tokenizer {name!r} was not found: {path}")
        tokenizers[name] = Tokenizer.from_file(str(path))

    started = perf_counter()
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

    print(f"Evaluating the next {eval_bytes:,} UTF-8 bytes", flush=True)
    evaluated = CorpusStats()
    texts = limited_text(
        dataset, text_column, eval_bytes, evaluated, progress_label="Evaluated"
    )
    token_counts = {name: 0 for name in tokenizers}
    batch: list[str] = []
    batch_bytes = 0
    scoring_started = perf_counter()

    def score_batch() -> None:
        if not batch:
            return
        for name, tokenizer in tokenizers.items():
            token_counts[name] += sum(len(encoding.ids) for encoding in tokenizer.encode_batch(batch))

    for text in texts:
        batch.append(text)
        batch_bytes += len(text.encode("utf-8"))
        if batch_bytes >= BATCH_BYTES:
            score_batch()
            batch.clear()
            batch_bytes = 0
    score_batch()

    if evaluated.bytes_used < eval_bytes - MAX_UTF8_BOUNDARY_SHORTFALL:
        raise ValueError("dataset ended before the requested evaluation size")
    scoring_seconds = perf_counter() - scoring_started
    total_seconds = perf_counter() - started
    print(
        f"Evaluated {evaluated.bytes_used:,} UTF-8 bytes from "
        f"{evaluated.documents_used:,} documents "
        f"(scoring {scoring_seconds:.1f}s; total {total_seconds:.1f}s)",
        flush=True,
    )

    results = {}
    for name, tokenizer in tokenizers.items():
        token_count = token_counts[name]
        if token_count == 0:
            raise ValueError(f"tokenizer {name!r} produced no tokens")
        vocab_size = tokenizer.get_vocab_size()
        token_id_bits = max(1, ceil(log2(vocab_size)))
        fixed_id_bits = token_count * token_id_bits
        results[name] = {
            "tokenizer_path": str(tokenizer_paths[name]),
            "vocab_size": vocab_size,
            "token_count": token_count,
            "bytes_per_token": evaluated.bytes_used / token_count,
            "tokens_per_byte": token_count / evaluated.bytes_used,
            "fixed_id_bits_per_token": token_id_bits,
            "fixed_id_bits_per_byte": fixed_id_bits / evaluated.bytes_used,
            "fixed_id_compression_ratio": (evaluated.bytes_used * 8) / fixed_id_bits,
        }

    summary = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "text_column": text_column,
        "skip_bytes_requested": skip_bytes,
        "skipped_utf8_bytes": skipped.bytes_used,
        "skipped_documents": skipped.documents_used,
        "eval_bytes_requested": eval_bytes,
        "evaluated_utf8_bytes": evaluated.bytes_used,
        "evaluated_documents": evaluated.documents_used,
        "scoring_seconds": scoring_seconds,
        "total_seconds": total_seconds,
        "tokenizers": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {output_dir / 'results.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", action="append", required=True, type=parse_tokenizer_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--skip", type=parse_size, default=DEFAULT_SKIP_BYTES)
    parser.add_argument("--size", type=parse_size, default=DEFAULT_EVAL_BYTES)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tokenizer_paths = dict(args.tokenizer)
    if len(tokenizer_paths) != len(args.tokenizer):
        parser.error("tokenizer names must be unique")
    try:
        evaluate_tokenizers(
            tokenizer_paths=tokenizer_paths,
            output_dir=args.output_dir,
            dataset_id=args.dataset,
            config=args.config,
            split=args.split,
            text_column=args.text_column,
            skip_bytes=args.skip,
            eval_bytes=args.size,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
