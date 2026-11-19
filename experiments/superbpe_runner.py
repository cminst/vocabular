"""Run resumable gated-SuperBPE training with durable resource telemetry."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Callable

from experiments.gated_superbpe import (
    FASTBOUNDLESSBPE_RESUME_PATCH,
    FASTBOUNDLESSBPE_REVISION,
    completed_model_summary,
    publish_completed_model,
    trainer_fingerprint,
)
from experiments.superbpe_resume import publish_checkpoints, seed_resume_from_model


def process_rss_kib(pid: int) -> int | None:
    """Read the trainer's current RSS without importing anything into its process."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_training(
    training_dir: Path,
    model_dir: Path,
    total_vocab_size: int,
    transition_vocab_size: int,
    status_path: Path,
    commit: Callable[[], None],
    seed_model_dir: Path | None = None,
) -> dict:
    """Keep the GIL-holding Rust trainer in a child and persist periodic progress."""
    cached = completed_model_summary(
        model_dir,
        model_type="gated_superbpe",
        total_vocab_size=total_vocab_size,
        transition_vocab_size=transition_vocab_size,
    )
    if cached is not None:
        print(f"Using completed model: {model_dir}", flush=True)
        return cached

    resume_dir = model_dir.with_name(f"{model_dir.name}.resume")
    if seed_model_dir is not None:
        seeded = seed_resume_from_model(
            source_model_dir=seed_model_dir,
            target_resume_dir=resume_dir,
            training_dir=training_dir,
            total_vocab_size=total_vocab_size,
            transition_vocab_size=transition_vocab_size,
            trainer_revision=FASTBOUNDLESSBPE_REVISION,
            trainer_patch=FASTBOUNDLESSBPE_RESUME_PATCH,
            trainer_fingerprint=trainer_fingerprint(),
        )
        if seeded:
            commit()
            print(f"Seeded resume checkpoint from {seed_model_dir}", flush=True)
    with TemporaryDirectory(prefix="superbpe-profile-") as scratch:
        staged_dir = Path(scratch) / "model"
        args = [
            sys.executable,
            "-m",
            "experiments.gated_superbpe",
            "train-gated-superbpe",
            "--training-dir",
            str(training_dir),
            "--output-dir",
            str(staged_dir),
            "--total-vocab-size",
            str(total_vocab_size),
            "--transition-vocab-size",
            str(transition_vocab_size),
            "--resume-dir",
            str(resume_dir),
            "--progress-interval",
            "256",
        ]
        started = monotonic()
        peak_rss_kib = 0
        committed_word = (resume_dir / "word.model").is_file()
        committed_checkpoints = {
            path.name for path in resume_dir.glob("tokenizer_*.model")
        } if resume_dir.is_dir() else set()
        last_commit = started
        process = subprocess.Popen(args)
        print(f"Trainer PID: {process.pid}", flush=True)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    returncode = None
                rss_kib = process_rss_kib(process.pid)
                if rss_kib is not None:
                    peak_rss_kib = max(peak_rss_kib, rss_kib)
                new_checkpoints = (
                    publish_checkpoints(staged_dir, resume_dir, committed_checkpoints)
                    if staged_dir.is_dir() and resume_dir.is_dir()
                    else []
                )
                word_ready = (resume_dir / "word.model").is_file()
                status = {
                    "state": "running" if returncode is None else (
                        "completed" if returncode == 0 else "failed"
                    ),
                    "elapsed_seconds": round(monotonic() - started, 1),
                    "current_rss_kib": rss_kib,
                    "sampled_peak_rss_kib": peak_rss_kib,
                    "word_model_ready": word_ready,
                    "checkpoint_files": sorted(committed_checkpoints | set(new_checkpoints)),
                    "total_vocab_size": total_vocab_size,
                    "transition_vocab_size": transition_vocab_size,
                    "trainer_returncode": returncode,
                }
                write_status(status_path, status)
                if (
                    (word_ready and not committed_word)
                    or new_checkpoints
                    or monotonic() - last_commit >= 60
                    or returncode is not None
                ):
                    commit()
                    committed_word = word_ready
                    committed_checkpoints.update(new_checkpoints)
                    last_commit = monotonic()
                    print(
                        f"Probe {status['elapsed_seconds']:.0f}s: "
                        f"RSS {rss_kib or 0:,} KiB, peak {peak_rss_kib:,} KiB, "
                        f"word ready={word_ready}, checkpoints={len(committed_checkpoints)}",
                        flush=True,
                    )
                if returncode is not None:
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, args)
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        summary = publish_completed_model(staged_dir, model_dir)
        status["model_dir"] = str(model_dir)
        status["training_seconds"] = summary["training_seconds"]
        status["word_phase_seconds"] = summary["word_phase_seconds"]
        status["super_phase_seconds"] = summary["super_phase_seconds"]
        status["trainer_peak_rss_kib"] = summary["peak_rss_kib"]
        write_status(status_path, status)
        commit()
        return summary
