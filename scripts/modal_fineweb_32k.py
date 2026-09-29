"""Train a 32k BPE tokenizer on 10GB of FineWeb text with Modal.

Plan: modal run scripts/modal_fineweb_32k.py
Run:  modal run scripts/modal_fineweb_32k.py --submit
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


@app.function(image=image, volumes={"/results": volume}, cpu=4, memory=16_384, timeout=86_400)
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
