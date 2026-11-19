"""Build a reproducible OLMo mix text sample from shared GPFS shards."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from experiments.olmo_corpus import SOURCE_WEIGHTS, documents


def cache_text(
    dataset_root: Path,
    output_dir: Path,
    text_bytes: int = 10_000_000_000,
    seed: int = 43,
) -> dict:
    """Write one NFC JSONL corpus, with source shares based on OLMo token counts."""
    if text_bytes < 1:
        raise ValueError("text_bytes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "train.jsonl"
    meta_path = output_dir / "train_meta.json"
    if text_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
        if (metadata.get("dataset") != "allenai/olmo-mix-1124"
            or metadata.get("byte_limit") != text_bytes
            or not text_bytes - 3 * len(SOURCE_WEIGHTS)
                <= metadata.get("cached_utf8_bytes", -1) <= text_bytes
            or metadata.get("sampling_seed") != seed
            or metadata.get("dataset_root") != str(dataset_root)):
            raise ValueError("existing OLMo text cache belongs to another setup")
        actual_digest = sha256()
        with text_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                actual_digest.update(chunk)
        if actual_digest.hexdigest() != metadata.get("text_sha256"):
            raise ValueError("cached OLMo text differs from its recorded SHA-256")
        print(f"Using completed OLMo text cache: {text_path}", flush=True)
        return metadata
    if text_path.exists() or meta_path.exists():
        raise ValueError("incomplete OLMo cache; inspect before retrying")

    temporary = text_path.with_suffix(".jsonl.part")
    digest = sha256()
    source_stats = {}
    total_docs = 0
    total_bytes = 0
    allocated = 0
    weight_total = sum(SOURCE_WEIGHTS.values())
    started = perf_counter()
    with temporary.open("wb") as output:
        for index, (source, weight) in enumerate(SOURCE_WEIGHTS.items()):
            budget = (text_bytes - allocated if index == len(SOURCE_WEIGHTS) - 1
                      else round(text_bytes * weight / weight_total))
            allocated += budget
            source_bytes = 0
            source_docs = 0
            for text in documents(dataset_root, source, seed=seed):
                remaining = budget - source_bytes
                encoded = text.encode("utf-8")
                if len(encoded) > remaining:
                    text = encoded[:remaining].decode("utf-8", errors="ignore")
                    encoded = text.encode("utf-8")
                if text:
                    line = (json.dumps({"text": text}, ensure_ascii=False,
                                       separators=(",", ":")) + "\n").encode("utf-8")
                    output.write(line)
                    digest.update(line)
                    source_bytes += len(encoded)
                    source_docs += 1
                if source_bytes >= budget - 3:
                    break
                if source_bytes // 500_000_000 > (source_bytes - len(encoded)) // 500_000_000:
                    print(f"{source}: {source_bytes:,}/{budget:,} text bytes", flush=True)
            if source_bytes < budget - 3:
                raise ValueError(f"{source} exhausted before its text quota")
            source_stats[source] = {"utf8_bytes": source_bytes, "documents": source_docs}
            total_bytes += source_bytes
            total_docs += source_docs
            print(f"{source}: {source_bytes:,} bytes, {source_docs:,} documents", flush=True)

    if total_bytes < text_bytes - 3 * len(SOURCE_WEIGHTS):
        raise ValueError("OLMo sample missed its requested text size")
    temporary.replace(text_path)
    metadata = {
        "format": "jsonl-text-v1",
        "dataset": "allenai/olmo-mix-1124",
        "config": "default",
        "split": "train",
        "text_column": "text",
        "normalization": "NFC",
        "dataset_root": str(dataset_root),
        "sampling_seed": seed,
        "sampling_method": "shuffled_shards_source_weighted",
        "source_weights_billions_of_tokens": SOURCE_WEIGHTS,
        "source_stats": source_stats,
        "byte_limit": text_bytes,
        "cached_utf8_bytes": total_bytes,
        "cached_documents": total_docs,
        "text_path": str(text_path),
        "text_sha256": digest.hexdigest(),
        "seconds": perf_counter() - started,
    }
    meta_tmp = meta_path.with_suffix(".json.part")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    meta_tmp.replace(meta_path)
    print(f"Saved {total_bytes:,} OLMo text bytes to {text_path}", flush=True)
    return metadata
