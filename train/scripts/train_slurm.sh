#!/bin/bash
################################## Slurm 옵션 ##################################
#SBATCH --job-name=yaicon-train
#SBATCH --partition=gigabyte_RTX6000ADA
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/thesol1/yaicon/logs/%j.out
#SBATCH --error=/home/thesol1/yaicon/logs/%j.err
################################################################################

set -euo pipefail

cd /home/thesol1/yaicon
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sft|dpo> [additional args passed to train script]" >&2
  exit 1
fi

TASK="$1"
shift

if [[ "${TASK}" == "dpo" ]]; then
  python src/train/dpo_train.py "$@"
elif [[ "${TASK}" == "sft" ]]; then
  python src/train/sft_train.py "$@"
else
  echo "Unknown TASK '${TASK}', expected 'sft' or 'dpo'." >&2
  exit 1
fi
