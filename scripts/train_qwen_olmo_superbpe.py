"""Train 100k gated SuperBPE additions to Qwen on a 10GB OLMo mix sample.

Run --stage cache first to sample shared OLMo 2 November 2024 shards into one
NFC JSONL file under --root. Source shares follow published OLMo token counts;
shard order is seeded, so the corpus is a reproducible shard sample rather
than a uniform sample of every document. The existing Qwen BPE is frozen.
Run --stage source to copy and audit its pinned tokenizer against this corpus,
then --stage train to learn 100,000 gated supermerges from scratch. The old
FineWeb checkpoints cannot be reused because the corpus differs. Training
checkpoints and status are stored under --root. No LM weights are trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("/gpfs/projects/zlab/qxli2/qwen-olmo-superbpe-100k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "source", "train"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/gpfs/datasets/olmo-mix-1124"))
    parser.add_argument("--text-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    if args.stage == "cache":
        from experiments.olmo_text_cache import cache_text

        result = cache_text(args.dataset_root, args.root / "training",
                            text_bytes=args.text_bytes, seed=args.seed)
    elif args.stage == "source":
        from experiments.qwen_superbpe import prepare_source

        source_json = args.root / "inputs/tokenizer.json"
        result = prepare_source(source_json, args.root / "source",
                                args.root / "training")
    else:
        from experiments.qwen_superbpe import run_training

        result = run_training(
            training_dir=args.root / "training",
            source_dir=args.root / "source",
            model_dir=args.root / "model",
            status_path=args.root / "status.json",
            commit=lambda: None,
            extra_tokens=100_000,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
