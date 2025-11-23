#!/bin/bash
################################## Slurm 옵션 ##################################
#SBATCH --job-name=aligning
#SBATCH --partition=gigabyte_A6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/thesol1/yaicon/logs/%j.out
#SBATCH --error=/home/thesol1/yaicon/logs/%j.err
################################################################################


cd /home/thesol1/yaicon
source .venv/bin/activate
python sft_dataset_builder.py \
  --provider hyperclova-local \
  --hf-model naver-hyperclovax/HyperCLOVAX-SEED-Think-14B \
  --hf-device cuda \
  --hf-dtype int8 \
  --output sft_dataset_seed.jsonl \
  --sleep 0
