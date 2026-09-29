<p align="center">
  <img height="120" src="./vocabular.svg" alt="vocabular Logo" />
</p>

---
Explorations into LM vocabularies.

## Train a tokenizer

Install the dependencies with `uv pip install -r requirements.txt`. From the repository root, run:

```bash
python -m scripts.train_tokenizer \
  --dataset HuggingFaceFW/fineweb \
  --split train --text-column text \
  --size 10MB --vocab-size 32000 \
  --output-dir outputs/fineweb-10mb
```

The size accepts `1MB`, `10MB`, `1GB`, `10GB`, or any other positive amount in `B`, `KB`, `MB`, or `GB` (decimal units). Set `--vocab-size` for each run; there is no default. Training streams the selected Hugging Face dataset and stops after at most that many UTF-8 text bytes. A final document may be cut at the limit. The output directory must be empty; it receives `tokenizer.json` and `run.json` with the dataset settings and actual amount used. Use `--config` when the dataset has multiple configurations; `--split` and `--text-column` default to `train` and `text`.

## Modal: FineWeb 10GB / 32k

Select the intended Modal profile, then inspect the run plan. The [Modal runbook](https://app.notion.com/p/38bcf1e145e981408e66c84f1d26a0ea) is the source for profile and volume procedures.

```bash
python -m modal run scripts/modal_fineweb_32k.py
```

To launch the remote training job, add `--submit`. The script streams the `sample-10BT` configuration of `HuggingFaceFW/fineweb` and saves `tokenizer.json` and `run.json` under `/fineweb-10gb-32k/` in the `vocabular-tokenizers` Modal volume. It refuses to overwrite an existing run directory. The training text cap is 10,000,000,000 UTF-8 bytes; the job requests 32 CPUs, 16 GiB of memory, and at most 24 hours. Download the two artifacts with `python -m modal volume get vocabular-tokenizers /fineweb-10gb-32k outputs` after the job completes.
