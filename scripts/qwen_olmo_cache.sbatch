#!/bin/bash
#SBATCH --job-name=qwen-olmo-cache
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
# CPU-only job: gpu-h200 is Tillicum's node pool; no GPU is requested.
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
root=/gpfs/projects/zlab/qxli2/qwen-olmo-superbpe-100k
cd "$root/code"
source /gpfs/scrubbed/qxli2/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m scripts.train_qwen_olmo_superbpe --stage cache --root "$root"
python -m scripts.train_qwen_olmo_superbpe --stage source --root "$root"
