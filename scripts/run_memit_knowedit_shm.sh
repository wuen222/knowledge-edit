#!/usr/bin/env bash
set -euo pipefail

mkdir -p /dev/shm/hf_datasets /dev/shm/tmp

export HF_DATASETS_CACHE=/dev/shm/hf_datasets
export TMPDIR=/dev/shm/tmp
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/transformers

cd /root/autodl-tmp/knowledge-edit

exec /root/miniconda3/envs/EasyEdit/bin/python scripts/run_sequential_edit.py \
  --model /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct \
  --method MEMIT \
  --hparams /root/autodl-tmp/EasyEdit/hparams/MEMIT/llama3-8b.yaml \
  --prepared-dir /root/autodl-tmp/data/prepared/zsre_knowedit \
  --checkpoints 0 1 10 50 100 \
  --max-edits 100 \
  --probe-limit 200 \
  --batch-size 1 \
  --hidden-batch-size 1 \
  --easyedit-root /root/autodl-tmp/EasyEdit \
  > /root/autodl-tmp/logs/memit_knowedit_zsre.log 2>&1
