#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  OMP_NUM_THREADS=$(( $(nproc) / 2 ))
  ENCODE_DEVICE=cpu
  DECODE_BATCH=8
  FORWARD_BATCH=32
  SAMPLE_BATCH=16
  GPUS=0,1
  MIN_FREE_MIB=20000
  PROMPTS=./experiments/images/coco30k_val2014.txt
)

echo "[1/2] SD3 — budget (leaves)"
env "${COMMON[@]}" \
  SAMPLER_CONFIG=sampler-leaves.json \
  MATCH=budget \
  OUT_ROOT="/hdd/mb1921/results/sd3/paired_eps0.8_match-budget" \
  bash experiments/images/sweep_sd3.sh

echo "[2/2] SD3 — verification (no leaves)"
env "${COMMON[@]}" \
  MATCH=verification \
  OUT_ROOT="/hdd/mb1921/results/sd3/paired_eps0.8_match-verification" \
  bash experiments/images/sweep_sd3.sh