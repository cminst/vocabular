#!/bin/bash
#SBATCH --job-name=qwen-olmo-100k
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
# CPU-only job: gpu-h200 is Tillicum's node pool; no GPU is requested.
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00

set -euo pipefail
root=/gpfs/projects/zlab/qxli2/qwen-olmo-superbpe-100k
cd "$root/code"
source /gpfs/scrubbed/qxli2/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export RAYON_NUM_THREADS=32
python -m scripts.train_qwen_olmo_superbpe --stage train --root "$root"
