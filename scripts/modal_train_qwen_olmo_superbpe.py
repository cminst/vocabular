"""Train 100k Qwen SuperBPE additions on the cached 10 GB OLMo sample.

The corpus is the exact seeded OLMo Mix 1124 sample prepared for the Tillicum
run. Upload train.jsonl, train_meta.json, and the pinned Qwen tokenizer.json
to vocabular-tokenizers:/superbpe/qwen35-2b-olmo-10gb-100k/{training,inputs}
before running this script. The prepare stage verifies the corpus SHA-256 and
audits exact Qwen token IDs on 1,000 documents. The train stage learns 100,000
new gated merges from Qwen's frozen BPE; it does not train model weights.

Use `MODAL_PROFILE=aurora python -m modal run --detach` with `--stage train
--submit` after preparation. Checkpoints and status remain on the volume, so a
later invocation resumes the same run.
"""

from hashlib import sha256
import json
import os
from pathlib import Path

import modal


ROOT = Path("/results/superbpe/qwen35-2b-olmo-10gb-100k")
VOLUME_NAME = "vocabular-tokenizers"
SOURCE_SHA256 = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
TRAINING_BYTES = 10_000_000_000
FASTBOUNDLESSBPE_REVISION = "c2dce3131093aa4c7fc595c91b15a861dea3094c"
PARALLEL_PATCH = Path("patches/fastboundlessbpe_parallel.patch")
RESUME_PATCH = Path("patches/fastboundlessbpe_resume.patch")
BUILD_FINGERPRINT = sha256(PARALLEL_PATCH.read_bytes() + RESUME_PATCH.read_bytes()).hexdigest()

app = modal.App("vocabular-qwen-olmo-superbpe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl", "git")
    .uv_pip_install("datasets>=3.0", "maturin>=1.0,<2.0", "tokenizers>=0.20")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain 1.88.0",
        "git clone https://github.com/kensho-technologies/fastboundlessbpe.git "
        f"/opt/fastboundlessbpe && cd /opt/fastboundlessbpe && git checkout {FASTBOUNDLESSBPE_REVISION}",
    )
    .add_local_file(PARALLEL_PATCH, "/opt/fastboundlessbpe_parallel.patch", copy=True)
    .add_local_file(RESUME_PATCH, "/opt/fastboundlessbpe_resume.patch", copy=True)
    .run_commands(
        "cd /opt/fastboundlessbpe && git apply /opt/fastboundlessbpe_parallel.patch",
        "cd /opt/fastboundlessbpe && git apply /opt/fastboundlessbpe_resume.patch",
        "export PATH=/root/.cargo/bin:$PATH && cd /opt/fastboundlessbpe && "
        "maturin build --release --out /opt/wheels",
        "pip install /opt/wheels/*.whl",
    )
    .env({"RAYON_NUM_THREADS": "32", "FASTBOUNDLESSBPE_BUILD_FINGERPRINT": BUILD_FINGERPRINT})
    .add_local_python_source("experiments")
)


@app.function(image=image, volumes={"/results": volume}, cpu=4, memory=16_384, timeout=3_600)
def prepare() -> dict:
    from experiments.qwen_superbpe import prepare_source
    from experiments.superbpe_resume import file_sha256

    training_dir = ROOT / "training"
    text_path = training_dir / "train.jsonl"
    metadata_path = training_dir / "train_meta.json"
    tokenizer_path = ROOT / "inputs/tokenizer.json"
    for path in (text_path, metadata_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("dataset") != "allenai/olmo-mix-1124"
        or metadata.get("normalization") != "NFC"
        or metadata.get("cached_utf8_bytes") != TRAINING_BYTES
        or metadata.get("sampling_seed") != 43
    ):
        raise ValueError("uploaded corpus metadata does not match this run")
    if file_sha256(text_path) != metadata.get("text_sha256"):
        raise ValueError("uploaded training text SHA-256 differs from its manifest")
    if file_sha256(tokenizer_path) != SOURCE_SHA256:
        raise ValueError("uploaded Qwen tokenizer is not the pinned source")
    metadata["text_path"] = str(text_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    source = prepare_source(tokenizer_path, ROOT / "source", training_dir)
    volume.commit()
    print(f"Verified {metadata['cached_utf8_bytes']:,} text bytes and exact Qwen IDs", flush=True)
    return {"corpus_sha256": metadata["text_sha256"], "source": source}


@app.function(image=image, volumes={"/results": volume}, cpu=32,
              memory=(131_072, 393_216), timeout=86_400)
def train() -> dict:
    from experiments.qwen_superbpe import run_training

    return run_training(
        training_dir=ROOT / "training",
        source_dir=ROOT / "source",
        model_dir=ROOT / "model",
        status_path=ROOT / "status.json",
        commit=volume.commit,
        extra_tokens=100_000,
    )


@app.local_entrypoint()
def main(stage: str = "all", submit: bool = False) -> None:
    if stage not in {"all", "prepare", "train"}:
        raise ValueError("stage must be all, prepare, or train")
    print(f"Dataset: OLMo Mix 1124; {TRAINING_BYTES:,} cached UTF-8 bytes")
    print("Source: Qwen/Qwen3.5-2B-Base; 100,000 new gated SuperBPE tokens")
    print(f"Output: {VOLUME_NAME}:{ROOT}")
    if not submit:
        print("Add --submit to run.")
        return
    if stage in {"all", "prepare"}:
        prepare.remote()
    if stage in {"all", "train"}:
        call = train.spawn()
        print(f"Training call: {call.object_id}")
