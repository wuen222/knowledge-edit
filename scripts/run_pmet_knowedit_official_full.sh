#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/autodl-tmp/cache/datasets /root/autodl-tmp/cache/tmp /root/autodl-tmp/logs

export HF_HOME=/root/autodl-tmp/cache
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/transformers
export TMPDIR=/root/autodl-tmp/cache/tmp
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /root/autodl-tmp/knowledge-edit

{
  date
  echo "Starting PMET official full run with cached 100k covariance stats"
  echo "Using persistent HF datasets cache: ${HF_DATASETS_CACHE}"
  echo "Using hparams: /root/autodl-tmp/EasyEdit/hparams/PMET/llama3-8b.yaml"
  exec /root/miniconda3/envs/EasyEdit/bin/python scripts/run_sequential_edit.py \
    --model /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct \
    --method PMET \
    --hparams /root/autodl-tmp/EasyEdit/hparams/PMET/llama3-8b.yaml \
    --prepared-dir /root/autodl-tmp/data/prepared/zsre_knowedit \
    --checkpoints 0 1 10 50 100 \
    --max-edits 100 \
    --probe-limit 200 \
    --batch-size 1 \
    --hidden-batch-size 1 \
    --easyedit-root /root/autodl-tmp/EasyEdit
} > /root/autodl-tmp/logs/pmet_knowedit_zsre_official_full.log 2>&1
