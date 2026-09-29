"""Evaluate selected FineWeb tokenizers on a held-out Modal stream.

From the repository root, install with `uv pip install -r requirements.txt`.
Select the intended Modal profile according to the Notion Modal runbook.

Run this file with `modal run` to print the plan. Add `--submit` to start the
remote evaluation. Select aliases with `--tokenizers`, for example `32k,1m`.
The defaults skip the first 10GB of the dataset stream and evaluate the next
1GB. Change them with `--skip` and `--size`, using B, KB, MB, or GB units.

Results are written to the vocabular-tokenizers volume under /evals/<label>/.
The result includes bytes per token, tokens per byte, and fixed-width token-ID
compression statistics. Fixed-width token-ID compression is a comparison
metric, not a serialized tokenizer file size.
"""

import os
from pathlib import Path

import modal

from experiments.train_tokenizer import parse_size


DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
VOLUME_NAME = "vocabular-tokenizers"
TOKENIZERS = {
    "32k": Path("/results/fineweb-10gb-32k/tokenizer.json"),
    "1m": Path("/results/fineweb-10gb-1m/tokenizer.json"),
}

app = modal.App("vocabular-fineweb-eval")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN")})
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("datasets>=3.0", "tokenizers>=0.20")
    .add_local_python_source("experiments")
)


def select_tokenizers(value: str) -> dict[str, Path]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("select at least one tokenizer alias")
    unknown = sorted(set(names) - TOKENIZERS.keys())
    if unknown:
        raise ValueError(f"unknown tokenizer aliases: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("tokenizer aliases must be unique")
    return {name: TOKENIZERS[name] for name in names}


@app.function(
    image=image,
    volumes={"/results": volume},
    secrets=[hf_secret],
    cpu=32,
    memory=16_384,
    timeout=86_400,
)
def evaluate(
    tokenizer_paths: dict[str, str], eval_bytes: int, skip_bytes: int, output_dir: str
) -> None:
    from experiments.tokenizer_eval_fineweb import evaluate_tokenizers

    evaluate_tokenizers(
        tokenizer_paths={name: Path(path) for name, path in tokenizer_paths.items()},
        output_dir=Path(output_dir),
        dataset_id=DATASET,
        config=CONFIG,
        skip_bytes=skip_bytes,
        eval_bytes=eval_bytes,
    )
    volume.commit()
    print(f"Committed evaluation to {VOLUME_NAME}:{output_dir}", flush=True)


@app.local_entrypoint()
def main(
    tokenizers: str = "32k,1m",
    size: str = "1GB",
    skip: str = "10GB",
    label: str = "fineweb-10gb-skip-1gb",
    submit: bool = False,
) -> None:
    tokenizer_paths = select_tokenizers(tokenizers)
    eval_bytes = parse_size(size)
    skip_bytes = parse_size(skip)
    output_dir = f"/results/evals/{label}"
    print(f"Dataset: {DATASET} ({CONFIG}, train/text)")
    print(f"Skipped prefix: {skip_bytes:,} UTF-8 bytes")
    print(f"Evaluation size: {eval_bytes:,} UTF-8 bytes")
    print(f"Tokenizers: {', '.join(tokenizer_paths)}")
    print(f"Output: {VOLUME_NAME}:{output_dir}")
    if submit:
        if not os.getenv("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN must be set locally before submitting")
        evaluate.remote(
            {name: str(path) for name, path in tokenizer_paths.items()},
            eval_bytes,
            skip_bytes,
            output_dir,
        )
