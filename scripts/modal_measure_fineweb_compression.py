"""Cache held-out FineWeb text and measure 1M BPE merge-prefix compression.

From the repository root, install with `uv pip install -r requirements.txt`.
Select the intended Modal profile according to the Notion Modal runbook.

Run this file with `modal run` to print the plan. Add `--submit` to start the
remote work. `--stage prepare` caches the held-out text, `--stage measure`
uses that cache, and `--stage all` (the default) does both. The data cache is
document-preserving JSONL, so every BPE prefix is evaluated on identical text.

The defaults skip the first 10GB of FineWeb, cache the next 1GB in the
vocabular-tokenizers volume, and measure every 16,000 vocabulary entries from
the 1M BPE tokenizer. This launcher writes JSON measurements only; use the
local plotting script after downloading the results.
"""

import os
from pathlib import Path

import modal

from experiments.train_tokenizer import parse_size


DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
VOLUME_NAME = "vocabular-tokenizers"
SOURCE_TOKENIZER = Path("/results/fineweb-10gb-1m/tokenizer.json")

app = modal.App("vocabular-fineweb-compression")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("datasets>=3.0", "tokenizers>=0.20")
    .add_local_python_source("experiments")
)


@app.function(
    image=image,
    volumes={"/results": volume},
    secrets=[hf_secret],
    cpu=4,
    memory=4_096,
    timeout=86_400,
)
def prepare(output_dir: str, skip_bytes: int, eval_bytes: int) -> None:
    from experiments.compression_vs_vocab_size import prepare_heldout_text

    prepare_heldout_text(
        output_dir=Path(output_dir),
        dataset_id=DATASET,
        config=CONFIG,
        skip_bytes=skip_bytes,
        eval_bytes=eval_bytes,
    )
    volume.commit()
    print(f"Committed held-out text to {VOLUME_NAME}:{output_dir}", flush=True)


@app.function(
    image=image,
    volumes={"/results": volume},
    cpu=32,
    memory=16_384,
    timeout=86_400,
)
def measure(output_dir: str, vocab_step: int) -> None:
    from experiments.compression_vs_vocab_size import measure_compression

    measure_compression(
        source_path=SOURCE_TOKENIZER,
        heldout_dir=Path(output_dir),
        output_path=Path(output_dir) / "compression.json",
        vocab_step=vocab_step,
    )
    volume.commit()
    print(f"Committed measurements to {VOLUME_NAME}:{output_dir}", flush=True)


@app.local_entrypoint()
def main(
    stage: str = "all",
    size: str = "1GB",
    skip: str = "10GB",
    vocab_step: int = 16_000,
    label: str = "fineweb-10gb-skip-1gb",
    submit: bool = False,
) -> None:
    if stage not in {"prepare", "measure", "all"}:
        raise ValueError("stage must be prepare, measure, or all")
    if vocab_step < 1:
        raise ValueError("vocab step must be positive")
    eval_bytes = parse_size(size)
    skip_bytes = parse_size(skip)
    output_dir = f"/results/evals/{label}"
    print(f"Dataset: {DATASET} ({CONFIG}, train/text)")
    print(f"Skipped prefix: {skip_bytes:,} UTF-8 bytes")
    print(f"Cached evaluation size: {eval_bytes:,} UTF-8 bytes")
    print(f"Source tokenizer: {VOLUME_NAME}:{SOURCE_TOKENIZER}")
    print(f"Vocabulary step: {vocab_step:,}")
    print(f"Output: {VOLUME_NAME}:{output_dir}")
    if submit:
        if stage in {"prepare", "all"} and not os.getenv("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN must be set locally before preparing")
        if stage in {"prepare", "all"}:
            prepare.remote(output_dir, skip_bytes, eval_bytes)
        if stage in {"measure", "all"}:
            measure.remote(output_dir, vocab_step)
