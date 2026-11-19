"""Prepare, train, and score native gated SuperBPE tokenizers."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shutil
from time import perf_counter
from uuid import uuid4

from datasets import load_dataset

from experiments.compression_vs_vocab_size import CACHE_FORMAT, MAX_UTF8_BOUNDARY_SHORTFALL
from experiments.train_tokenizer import CorpusStats, limited_text


SPECIAL_TOKENS = ["<|endoftext|>"]
PROGRESS_BYTES = 100_000_000
MERGE_LOG_INTERVAL = 8_192
# recalc only performs full-state audits; it does not update training counts.
VERIFY_INTERVAL = 0
FASTBOUNDLESSBPE_REVISION = "c2dce3131093aa4c7fc595c91b15a861dea3094c"
FASTBOUNDLESSBPE_PATCH = "patches/fastboundlessbpe_parallel.patch"
FASTBOUNDLESSBPE_RESUME_PATCH = "patches/fastboundlessbpe_resume.patch"


def trainer_fingerprint() -> str:
    """Identify the exact Rust patches used by this trainer build."""
    fingerprint = os.getenv("FASTBOUNDLESSBPE_BUILD_FINGERPRINT")
    if fingerprint:
        return fingerprint
    repository_root = Path(__file__).resolve().parent.parent
    digest = sha256()
    for patch_name in (FASTBOUNDLESSBPE_PATCH, FASTBOUNDLESSBPE_RESUME_PATCH):
        digest.update((repository_root / patch_name).read_bytes())
    return digest.hexdigest()


def completed_model_summary(
    output_dir: Path,
    model_type: str,
    total_vocab_size: int,
    transition_vocab_size: int | None,
) -> dict | None:
    """Return a matching completed model summary, rejecting partial outputs."""
    if not output_dir.exists():
        return None
    run_path = output_dir / "run.json"
    model_path = output_dir / "tokenizer.model"
    if not run_path.is_file() or not model_path.is_file():
        raise ValueError(f"model output directory is incomplete: {output_dir}")
    summary = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        summary.get("model_type") != model_type
        or summary.get("total_vocab_size_requested") != total_vocab_size
        or summary.get("transition_vocab_size_requested") != transition_vocab_size
    ):
        raise ValueError(f"model output directory does not match this run: {output_dir}")
    return summary


def publish_completed_model(staged_dir: Path, output_dir: Path) -> dict:
    """Atomically publish the final model files from local scratch to a volume."""
    run_path = staged_dir / "run.json"
    model_path = staged_dir / "tokenizer.model"
    if not run_path.is_file() or not model_path.is_file():
        raise ValueError(f"staged model is incomplete: {staged_dir}")
    if output_dir.exists():
        raise ValueError(f"model output directory already exists: {output_dir}")
    summary = json.loads(run_path.read_text(encoding="utf-8"))
    summary["tokenizer_path"] = str(output_dir / "tokenizer.model")
    staged_word_model = staged_dir / "word.model"
    if staged_word_model.is_file():
        summary["word_model_path"] = str(output_dir / "word.model")
    run_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    temporary_dir = output_dir.parent / f".{output_dir.name}.publishing-{uuid4().hex}"
    temporary_dir.mkdir(parents=True)
    for filename in ("tokenizer.model", "word.model", "run.json"):
        source = staged_dir / filename
        if source.is_file():
            shutil.copy2(source, temporary_dir / filename)
    temporary_dir.replace(output_dir)
    return summary


def cache_training_text(
    output_dir: Path,
    dataset_id: str,
    config: str | None,
    split: str,
    text_column: str,
    size: int,
    source_dir: Path | None = None,
) -> dict:
    """Materialize a bounded streaming dataset prefix as JSONL for native training."""
    text_path = output_dir / "train.jsonl"
    metadata_path = output_dir / "train_meta.json"
    temporary_path = output_dir / "train.jsonl.part"
    if text_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != CACHE_FORMAT:
            raise ValueError(f"training cache has an unsupported format: {text_path}")
        if metadata.get("byte_limit") != size:
            raise ValueError(f"training cache has a different byte limit: {text_path}")
        print(f"Using cached training text: {text_path}", flush=True)
        return metadata
    if text_path.exists() or metadata_path.exists():
        raise ValueError(f"incomplete training text cache in {output_dir}")
    if temporary_path.exists():
        print(f"Discarding interrupted cache write: {temporary_path}", flush=True)
        temporary_path.unlink()
    if size < 1:
        raise ValueError("training byte limit must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path: Path | None = None
    if source_dir is not None:
        source_path = source_dir / "train.jsonl"
        source_metadata_path = source_dir / "train_meta.json"
        if not source_path.is_file() or not source_metadata_path.is_file():
            raise ValueError(f"source training cache is incomplete: {source_dir}")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if any(
            source_metadata.get(key) != value
            for key, value in {
                "format": CACHE_FORMAT,
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "text_column": text_column,
            }.items()
        ) or source_metadata.get("cached_utf8_bytes", 0) < size:
            raise ValueError(f"source training cache does not cover this prefix: {source_dir}")
        print(f"Reading prefix from cached corpus: {source_path}", flush=True)

        def cached_rows():
            with source_path.open("r", encoding="utf-8") as source:
                for line in source:
                    yield json.loads(line)

        rows = cached_rows()
    else:
        print(
            f"Opening streamed dataset: {dataset_id} "
            f"({config or 'default'}, {split}/{text_column})",
            flush=True,
        )
        rows = load_dataset(dataset_id, name=config, split=split, streaming=True)
    print(f"Caching the first {size:,} UTF-8 bytes for training", flush=True)
    cached = CorpusStats()
    with temporary_path.open("w", encoding="utf-8") as handle:
        for text in limited_text(
            rows, text_column, size, cached, progress_label="Cached"
        ):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    if cached.bytes_used < size - MAX_UTF8_BOUNDARY_SHORTFALL:
        raise ValueError("dataset ended before the requested training prefix")
    temporary_path.replace(text_path)
    metadata = {
        "format": CACHE_FORMAT,
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "text_column": text_column,
        "byte_limit": size,
        "cached_utf8_bytes": cached.bytes_used,
        "cached_documents": cached.documents_used,
        "text_path": str(text_path),
        "source_cache": str(source_path) if source_path is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved training text to {text_path}", flush=True)
    return metadata


def train_gated_superbpe(
    training_dir: Path,
    output_dir: Path,
    total_vocab_size: int,
    transition_vocab_size: int,
    resume_dir: Path | None = None,
    progress_interval: int = MERGE_LOG_INTERVAL,
) -> dict:
    """Train one two-pass gated SuperBPE model using the native Rust trainers."""
    metadata_path = training_dir / "train_meta.json"
    text_path = training_dir / "train.jsonl"
    if not text_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"training cache is missing from {training_dir}")
    if not 1 < transition_vocab_size < total_vocab_size:
        raise ValueError("transition vocabulary size must be between one and total size")
    if progress_interval < 1:
        raise ValueError("progress interval must be positive")
    cached = completed_model_summary(
        output_dir,
        model_type="gated_superbpe",
        total_vocab_size=total_vocab_size,
        transition_vocab_size=transition_vocab_size,
    )
    if cached is not None:
        print(f"Using cached gated SuperBPE model: {output_dir / 'tokenizer.model'}", flush=True)
        return cached

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError("training cache must use JSONL text records")
    if resume_dir is not None:
        from experiments.superbpe_resume import prepare_resume_dir

        print(f"Validating resume corpus and config: {resume_dir}", flush=True)
        prepare_resume_dir(
            resume_dir,
            metadata,
            text_path,
            total_vocab_size,
            transition_vocab_size,
            FASTBOUNDLESSBPE_REVISION,
            FASTBOUNDLESSBPE_RESUME_PATCH,
            trainer_fingerprint(),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    from boundlessbpe import (
        DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
        FastBpeTrainer,
        FastSuperBpeTrainer,
        FastTokenizer,
        SCRIPT_SPECIFIC_GPT4O_REGEX,
    )

    if FastBpeTrainer is None or FastSuperBpeTrainer is None or FastTokenizer is None:
        raise RuntimeError("fastboundlessbpe Rust bindings are unavailable")

    documents = metadata["cached_documents"]
    byte_limit = metadata["cached_utf8_bytes"]
    supermerge_count = total_vocab_size - transition_vocab_size
    staged_word_path = output_dir / "word.model"
    staged_model_path = output_dir / "tokenizer.model"
    word_prefix = output_dir / "word"
    model_prefix = output_dir / "tokenizer"
    word_model_path = (resume_dir / "word.model") if resume_dir else staged_word_path
    trainer_options = {
        "script_specific_regex": SCRIPT_SPECIFIC_GPT4O_REGEX,
        "script_specific_scripts": DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
    }
    started = perf_counter()
    print(
        f"Training gated SuperBPE: total={total_vocab_size:,}, "
        f"transition={transition_vocab_size:,}, supermerges={supermerge_count:,}",
        flush=True,
    )
    if word_model_path.is_file():
        print(f"Using cached word model: {word_model_path}", flush=True)
    else:
        word_trainer = FastBpeTrainer(**trainer_options)
        word_trainer.train(
            tau=1.1,
            filepath=str(text_path),
            outprefix=str(word_prefix),
            num_lines=documents,
            vocab_size=transition_vocab_size,
            recalc=VERIFY_INTERVAL,
            blowup=False,
            max_bytes=byte_limit,
            checkpoint_iterations=0,
            verbose=True,
            progress_interval=MERGE_LOG_INTERVAL,
            save_pretokens=None,
        )
        word_trainer.register_special_tokens(SPECIAL_TOKENS)
        word_trainer.save(str(word_prefix))
        if resume_dir is not None:
            from experiments.superbpe_resume import publish_word_model

            word_model_path = publish_word_model(staged_word_path, resume_dir)

    word_tokenizer = FastTokenizer()
    word_tokenizer.load(str(word_model_path))
    word_vocab_size_actual = word_tokenizer.get_vocab_size(False)
    if word_vocab_size_actual != transition_vocab_size:
        raise ValueError(
            f"word trainer reached {word_vocab_size_actual} tokens, "
            f"not the requested {transition_vocab_size}"
        )
    word_phase_seconds = perf_counter() - started
    word_phase_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    resume_options: dict = {}
    resumed_from_supermerges = 0
    checkpoint_interval = 0
    if resume_dir is not None:
        from experiments.superbpe_resume import CHECKPOINT_INTERVAL, latest_checkpoint

        resume_model_path, resumed_from_supermerges = latest_checkpoint(
            resume_dir, transition_vocab_size, total_vocab_size
        )
        checkpoint_interval = CHECKPOINT_INTERVAL
        if resume_model_path is not None:
            resume_options["resume_model_file"] = str(resume_model_path)
            print(f"Resuming from {resume_model_path}", flush=True)
    super_trainer = FastSuperBpeTrainer(**trainer_options)
    super_trainer.train(
        filepath=str(text_path),
        outprefix=str(model_prefix),
        num_lines=documents,
        vocab_size=supermerge_count,
        recalc=VERIFY_INTERVAL,
        word_model_file=str(word_model_path),
        max_bytes=byte_limit,
        checkpoint_iterations=checkpoint_interval,
        verbose=True,
        progress_interval=progress_interval,
        save_pretokens=None,
        greedy_split=False,
        min_count=15,
        max_ngram_len=30,
        **resume_options,
    )
    super_trainer.register_special_tokens(SPECIAL_TOKENS)
    super_trainer.save(str(model_prefix))
    super_phase_seconds = perf_counter() - started - word_phase_seconds
    if resume_dir is not None:
        shutil.copy2(word_model_path, staged_word_path)

    tokenizer = FastTokenizer()
    tokenizer.load(str(staged_model_path))
    actual_vocab_size = tokenizer.get_vocab_size(False)
    probe = "Native gated SuperBPE round-trip."
    if tokenizer.decode(tokenizer.encode_ordinary(probe)) != probe:
        raise ValueError("native gated SuperBPE failed a decode round-trip")
    summary = {
        "model_type": "gated_superbpe",
        "implementation": {
            "repository": "https://github.com/kensho-technologies/fastboundlessbpe",
            "revision": FASTBOUNDLESSBPE_REVISION,
            "patch": FASTBOUNDLESSBPE_PATCH,
            "resume_patch": FASTBOUNDLESSBPE_RESUME_PATCH,
            "fingerprint": trainer_fingerprint(),
        },
        "training_cache": metadata,
        "total_vocab_size_requested": total_vocab_size,
        "transition_vocab_size_requested": transition_vocab_size,
        "supermerge_count_requested": supermerge_count,
        "special_tokens": SPECIAL_TOKENS,
        "pretokenizer": "script-specific-gpt4o",
        "word_model_path": str(staged_word_path),
        "tokenizer_path": str(staged_model_path),
        "vocab_size_actual": actual_vocab_size,
        "word_vocab_size_actual": word_vocab_size_actual,
        "supermerge_count_actual": actual_vocab_size - word_vocab_size_actual,
        "resumed_from_supermerges": resumed_from_supermerges,
        "checkpoint_interval": checkpoint_interval,
        "resume_dir": str(resume_dir) if resume_dir is not None else None,
        "word_phase_seconds": word_phase_seconds,
        "super_phase_seconds": super_phase_seconds,
        "word_phase_peak_rss_kib": word_phase_peak_rss_kib,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_seconds": perf_counter() - started,
    }
    (output_dir / "run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_native_bpe(
    training_dir: Path,
    output_dir: Path,
    total_vocab_size: int,
) -> dict:
    """Train a native word-level BPE control with the same pretokenizer."""
    metadata_path = training_dir / "train_meta.json"
    text_path = training_dir / "train.jsonl"
    if not text_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"training cache is missing from {training_dir}")
    cached = completed_model_summary(
        output_dir,
        model_type="native_bpe",
        total_vocab_size=total_vocab_size,
        transition_vocab_size=None,
    )
    if cached is not None:
        print(f"Using cached native BPE model: {output_dir / 'tokenizer.model'}", flush=True)
        return cached
    run_path = output_dir / "run.json"
    model_path = output_dir / "tokenizer.model"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError("training cache must use JSONL text records")
    output_dir.mkdir(parents=True, exist_ok=True)
    from boundlessbpe import (
        DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
        FastBpeTrainer,
        FastTokenizer,
        SCRIPT_SPECIFIC_GPT4O_REGEX,
    )

    if FastBpeTrainer is None or FastTokenizer is None:
        raise RuntimeError("fastboundlessbpe Rust bindings are unavailable")
    started = perf_counter()
    trainer = FastBpeTrainer(
        script_specific_regex=SCRIPT_SPECIFIC_GPT4O_REGEX,
        script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
    )
    print(f"Training native BPE control: total={total_vocab_size:,}", flush=True)
    trainer.train(
        tau=1.1,
        filepath=str(text_path),
        outprefix=str(output_dir / "tokenizer"),
        num_lines=metadata["cached_documents"],
        vocab_size=total_vocab_size,
        recalc=VERIFY_INTERVAL,
        blowup=False,
        max_bytes=metadata["cached_utf8_bytes"],
        checkpoint_iterations=0,
        verbose=True,
        progress_interval=MERGE_LOG_INTERVAL,
        save_pretokens=None,
    )
    trainer.register_special_tokens(SPECIAL_TOKENS)
    trainer.save(str(output_dir / "tokenizer"))
    tokenizer = FastTokenizer()
    tokenizer.load(str(model_path))
    actual_vocab_size = tokenizer.get_vocab_size(False)
    probe = "Native BPE round-trip."
    if tokenizer.decode(tokenizer.encode_ordinary(probe)) != probe:
        raise ValueError("native BPE failed a decode round-trip")
    summary = {
        "model_type": "native_bpe",
        "implementation": {
            "repository": "https://github.com/kensho-technologies/fastboundlessbpe",
            "revision": FASTBOUNDLESSBPE_REVISION,
            "patch": FASTBOUNDLESSBPE_PATCH,
        },
        "training_cache": metadata,
        "total_vocab_size_requested": total_vocab_size,
        "transition_vocab_size_requested": None,
        "supermerge_count_requested": 0,
        "special_tokens": SPECIAL_TOKENS,
        "pretokenizer": "script-specific-gpt4o",
        "tokenizer_path": str(model_path),
        "vocab_size_actual": actual_vocab_size,
        "word_vocab_size_actual": actual_vocab_size,
        "supermerge_count_actual": 0,
        "training_seconds": perf_counter() - started,
    }
    run_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def score_native_model(model_dir: Path, heldout_dir: Path) -> dict:
    """Score a native BPE-family model on JSONL held-out text without mutation."""
    model_path = model_dir / "tokenizer.model"
    run_path = model_dir / "run.json"
    text_path = heldout_dir / "heldout.jsonl"
    metadata_path = heldout_dir / "heldout_meta.json"
    if not model_path.is_file() or not run_path.is_file():
        raise ValueError(f"model is missing from {model_dir}")
    if not text_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"held-out cache is missing from {heldout_dir}")

    model = json.loads(run_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError("held-out cache must use JSONL text records")

    from boundlessbpe import FastTokenizer

    if FastTokenizer is None:
        raise RuntimeError("fastboundlessbpe Rust bindings are unavailable")
    tokenizer = FastTokenizer()
    tokenizer.load(str(model_path))
    started = perf_counter()
    token_count = 0
    bytes_scored = 0
    next_progress = PROGRESS_BYTES
    batch: list[str] = []

    def score_batch() -> int:
        if not batch:
            return 0
        return sum(len(tokens) for tokens in tokenizer.encode_batch(batch))

    with text_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"invalid JSONL text record in {text_path}")
            batch.append(text)
            bytes_scored += len(text.encode("utf-8"))
            if len(batch) == 1_024:
                token_count += score_batch()
                batch.clear()
            if bytes_scored >= next_progress:
                print(
                    f"Scored {bytes_scored:,} / {metadata['cached_utf8_bytes']:,} UTF-8 bytes",
                    flush=True,
                )
                next_progress = ((bytes_scored // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
    token_count += score_batch()
    if bytes_scored != metadata["cached_utf8_bytes"]:
        raise ValueError(
            f"held-out byte count changed: {bytes_scored:,} != "
            f"{metadata['cached_utf8_bytes']:,}"
        )
    if token_count == 0:
        raise ValueError("native tokenizer produced no tokens")
    result = {
        "label": model_dir.name,
        "model_type": model["model_type"],
        "total_vocab_size_requested": model["total_vocab_size_requested"],
        "transition_vocab_size_requested": model["transition_vocab_size_requested"],
        "supermerge_count_requested": model["supermerge_count_requested"],
        "vocab_size_actual": model["vocab_size_actual"],
        "word_vocab_size_actual": model["word_vocab_size_actual"],
        "supermerge_count_actual": model["supermerge_count_actual"],
        "tokenizer_path": str(model_path),
        "token_count": token_count,
        "bytes_per_token": bytes_scored / token_count,
        "tokens_per_byte": token_count / bytes_scored,
        "scoring_seconds": perf_counter() - started,
    }
    print(
        f"{result['label']}: {result['bytes_per_token']:.4f} bytes/token "
        f"({result['scoring_seconds']:.1f}s)",
        flush=True,
    )
    return result


def write_pilot_results(
    output_path: Path,
    training_dir: Path,
    heldout_dir: Path,
    points: list[dict],
) -> dict:
    """Write one immutable, sorted result file for the completed pilot."""
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("model_type") != "gated_superbpe":
            raise ValueError(f"output already exists with a different format: {output_path}")
        print(f"Using cached gated SuperBPE results: {output_path}", flush=True)
        return existing
    if not points:
        raise ValueError("pilot produced no evaluation points")
    training_metadata = json.loads(
        (training_dir / "train_meta.json").read_text(encoding="utf-8")
    )
    heldout_metadata = json.loads(
        (heldout_dir / "heldout_meta.json").read_text(encoding="utf-8")
    )
    points.sort(
        key=lambda point: (
            point["total_vocab_size_requested"],
            point["transition_vocab_size_requested"] is not None,
            point["transition_vocab_size_requested"] or 0,
        )
    )
    summary = {
        "model_type": "gated_superbpe",
        "training_cache": training_metadata,
        "heldout_cache": heldout_metadata,
        "points": points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved gated SuperBPE results to {output_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train-bpe", "train-gated-superbpe"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--training-dir", required=True, type=Path)
        command_parser.add_argument("--output-dir", required=True, type=Path)
        command_parser.add_argument("--total-vocab-size", required=True, type=int)
        if command == "train-gated-superbpe":
            command_parser.add_argument("--transition-vocab-size", required=True, type=int)
            command_parser.add_argument("--resume-dir", type=Path)
            command_parser.add_argument(
                "--progress-interval", type=int, default=MERGE_LOG_INTERVAL
            )
    args = parser.parse_args()
    if args.command == "train-bpe":
        train_native_bpe(args.training_dir, args.output_dir, args.total_vocab_size)
    else:
        train_gated_superbpe(
            args.training_dir,
            args.output_dir,
            args.total_vocab_size,
            args.transition_vocab_size,
            args.resume_dir,
            args.progress_interval,
        )


if __name__ == "__main__":
    main()
