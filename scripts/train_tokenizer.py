"""Train a tokenizer on a byte-limited Hugging Face dataset stream.

From the repository root, install with `uv pip install -r requirements.txt`
and run this module with arguments such as:
    --dataset HuggingFaceFW/fineweb --config sample-10BT
    --size 10MB --vocab-size 32000
    --output-dir outputs/fineweb-10mb

Sizes use decimal B, KB, MB, or GB. Dataset rows are read in stream order,
using the `text` column of the `train` split unless overridden. The limit
counts UTF-8 text bytes, and the final document may be truncated. The output
directory must be empty; it receives tokenizer.json and run.json.
"""

from experiments.train_tokenizer import main


if __name__ == "__main__":
    main()
