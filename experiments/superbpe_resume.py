"""Validate and locate durable gated-SuperBPE resume artifacts."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil


RESUME_FORMAT = "gated-superbpe-resume-v1"
CHECKPOINT_INTERVAL = 8_192
CHECKPOINT_NAME = re.compile(r"tokenizer_(\d+)\.model\Z")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resume_dir(
    resume_dir: Path,
    training_metadata: dict,
    text_path: Path,
    total_vocab_size: int,
    transition_vocab_size: int,
    trainer_revision: str,
    trainer_patch: str,
    trainer_fingerprint: str,
) -> None:
    """Bind a checkpoint directory to this exact corpus and training setup."""
    expected = {
        "format": RESUME_FORMAT,
        "training_cache": training_metadata,
        "training_text_sha256": file_sha256(text_path),
        "total_vocab_size": total_vocab_size,
        "transition_vocab_size": transition_vocab_size,
        "trainer_revision": trainer_revision,
        "trainer_patch": trainer_patch,
        "trainer_fingerprint": trainer_fingerprint,
    }
    manifest_path = resume_dir / "manifest.json"
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"resume directory belongs to a different run: {resume_dir}")
        return
    if resume_dir.exists() and any(
        not path.name.endswith(".part") for path in resume_dir.iterdir()
    ):
        raise ValueError(f"resume directory has no manifest: {resume_dir}")
    resume_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = resume_dir / "manifest.json.part"
    temporary_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, manifest_path)


def publish_word_model(source: Path, resume_dir: Path) -> Path:
    """Make the completed word model visible under one atomic filename."""
    destination = resume_dir / "word.model"
    temporary_path = resume_dir / "word.model.part"
    shutil.copy2(source, temporary_path)
    os.replace(temporary_path, destination)
    return destination


def seed_resume_from_model(
    source_model_dir: Path,
    target_resume_dir: Path,
    training_dir: Path,
    total_vocab_size: int,
    transition_vocab_size: int,
    trainer_revision: str,
    trainer_patch: str,
    trainer_fingerprint: str,
) -> bool:
    """Seed a larger run from a completed smaller run on the identical corpus."""
    source_run_path = source_model_dir / "run.json"
    source_resume_dir = source_model_dir.with_name(f"{source_model_dir.name}.resume")
    source_manifest_path = source_resume_dir / "manifest.json"
    metadata_path = training_dir / "train_meta.json"
    text_path = training_dir / "train.jsonl"
    for path in (source_run_path, source_manifest_path, metadata_path, text_path):
        if not path.is_file():
            raise ValueError(f"resume seed is missing: {path}")

    summary = json.loads(source_run_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_vocab_size = summary.get("vocab_size_actual")
    if (
        summary.get("model_type") != "gated_superbpe"
        or not isinstance(source_vocab_size, int)
        or not transition_vocab_size < source_vocab_size < total_vocab_size
        or summary.get("transition_vocab_size_requested") != transition_vocab_size
        or summary.get("training_cache") != training_metadata
        or summary.get("implementation", {}).get("revision") != trainer_revision
        or summary.get("implementation", {}).get("fingerprint") != trainer_fingerprint
    ):
        raise ValueError(f"source model is incompatible with this run: {source_model_dir}")

    prepare_resume_dir(
        target_resume_dir,
        training_metadata,
        text_path,
        total_vocab_size,
        transition_vocab_size,
        trainer_revision,
        trainer_patch,
        trainer_fingerprint,
    )
    target_manifest = json.loads(
        (target_resume_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if any(
        source_manifest.get(key) != target_manifest[key]
        for key in target_manifest.keys() - {"total_vocab_size"}
    ) or source_manifest.get("total_vocab_size") != source_vocab_size:
        raise ValueError(f"source checkpoint manifest is incompatible: {source_resume_dir}")

    changed = False
    for filename in ("word.model", f"tokenizer_{source_vocab_size}.model"):
        source = source_resume_dir / filename
        destination = target_resume_dir / filename
        if not source.is_file():
            raise ValueError(f"resume seed is missing: {source}")
        if destination.is_file():
            if file_sha256(source) != file_sha256(destination):
                raise ValueError(f"resume seed conflicts with existing file: {destination}")
            continue
        temporary_path = target_resume_dir / f"{filename}.part"
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
        changed = True
    return changed


def publish_checkpoints(
    staged_dir: Path, resume_dir: Path, published: set[str]
) -> list[str]:
    """Copy only complete, atomically renamed Rust snapshots to durable storage."""
    new_names: list[str] = []
    for source in sorted(staged_dir.iterdir()):
        if CHECKPOINT_NAME.fullmatch(source.name) is None or source.name in published:
            continue
        destination = resume_dir / source.name
        if destination.exists():
            if file_sha256(source) != file_sha256(destination):
                raise ValueError(f"checkpoint name has conflicting contents: {destination}")
        else:
            temporary_path = resume_dir / f"{source.name}.part"
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, destination)
        new_names.append(source.name)
    return new_names


def latest_checkpoint(
    resume_dir: Path, transition_vocab_size: int, total_vocab_size: int
) -> tuple[Path | None, int]:
    """Select the highest complete checkpoint; ignore .part files."""
    best_path: Path | None = None
    best_size = transition_vocab_size
    for path in resume_dir.iterdir():
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        vocab_size = int(match.group(1))
        if not path.is_file() or not transition_vocab_size <= vocab_size <= total_vocab_size:
            raise ValueError(f"invalid SuperBPE checkpoint: {path}")
        if vocab_size > best_size:
            best_path, best_size = path, vocab_size
    return best_path, best_size - transition_vocab_size
