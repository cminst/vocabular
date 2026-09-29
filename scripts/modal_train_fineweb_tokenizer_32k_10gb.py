"""Train a 32k BPE tokenizer on 10GB of FineWeb text with Modal.

From the repository root, install with `uv pip install -r requirements.txt`.
Select the intended Modal profile according to the Notion Modal runbook.

Run this file with `modal run` to print the plan. Add `--submit` to start
remote training.

Streams HuggingFaceFW/fineweb, config sample-10BT, in dataset order. The
training cap is 10,000,000,000 UTF-8 text bytes. The function requests 32 CPUs,
16 GiB of memory, and a 24-hour timeout. It writes tokenizer.json and run.json
to the vocabular-tokenizers volume at /fineweb-10gb-32k/ and commits the volume.
An existing run directory is never overwritten.

Download after completion:
    python -m modal volume get vocabular-tokenizers /fineweb-10gb-32k outputs
"""

from pathlib import Path

import modal


DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
BYTE_LIMIT = 10_000_000_000
VOCAB_SIZE = 32_000
VOLUME_NAME = "vocabular-tokenizers"
OUTPUT_DIR = Path("/results/fineweb-10gb-32k")

app = modal.App("vocabular-fineweb-32k")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("datasets>=3.0", "tokenizers>=0.20")
    .add_local_python_source("experiments")
)


@app.function(image=image, volumes={"/results": volume}, cpu=32, memory=16_384, timeout=86_400)
def train() -> None:
    from experiments.train_tokenizer import train_tokenizer

    train_tokenizer(
        dataset_id=DATASET,
        config=CONFIG,
        split="train",
        text_column="text",
        size=BYTE_LIMIT,
        vocab_size=VOCAB_SIZE,
        output_dir=OUTPUT_DIR,
    )
    volume.commit()
    print(f"Committed tokenizer to {VOLUME_NAME}:{OUTPUT_DIR}", flush=True)


@app.local_entrypoint()
def main(submit: bool = False) -> None:
    print(f"Dataset: {DATASET} ({CONFIG}, train/text)")
    print(f"Corpus cap: {BYTE_LIMIT:,} UTF-8 bytes")
    print(f"Vocabulary size: {VOCAB_SIZE:,}")
    print(f"Output: {VOLUME_NAME}:{OUTPUT_DIR}")
    if submit:
        train.remote()
