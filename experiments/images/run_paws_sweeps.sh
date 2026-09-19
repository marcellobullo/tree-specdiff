#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] FFHQ — budget matching"
NETWORK="$PWD/edm/edm-ffhq-64x64-uncond-vp.pkl" \
DATASET=ffhq \
SAMPLER_CONFIG=sampler-leaves.json \
OUT_ROOT="$PWD/results/edm/ffhq/ffhq_n500_leaves_match-budget" \
NUM_SAMPLES=500 \
EPS="0.1 0.3 0.6" \
CONFIGS="2,3 2,5 3,4 4,4" \
GPUS=0,1 \
MATCH=budget \
NODE_BUDGET=100 \
RULES=paws \
bash experiments/images/sweep.sh

echo "[2/4] FFHQ — verification matching"
NETWORK="$PWD/edm/edm-ffhq-64x64-uncond-vp.pkl" \
DATASET=ffhq \
OUT_ROOT="$PWD/results/edm/ffhq/ffhq_n500_leaves_match-verification" \
NUM_SAMPLES=500 \
EPS="0.1 0.3 0.6" \
CONFIGS="2,3 2,5 3,4 4,4" \
GPUS=0,1 \
MATCH=verification \
NODE_BUDGET=100 \
RULES=paws \
bash experiments/images/sweep.sh

echo "[3/4] CIFAR-10 — budget matching"
NETWORK="$PWD/edm/edm-cifar10-32x32-cond-vp.pkl" \
DATASET=cifar10 \
NUM_SAMPLES=500 \
NODE_BUDGET=500 \
GPUS=0,1 \
EPS="0.1 0.3 0.6" \
MATCH=budget \
SAMPLER_CONFIG=experiments/images/sampler_leaves.json \
OUT_ROOT="$PWD/results/edm/cifar10/cifar10_n500_leaves_match-budget" \
RULES=paws \
bash experiments/images/sweep.sh

echo "[4/4] CIFAR-10 — verification matching"
NETWORK="$PWD/edm/edm-cifar10-32x32-cond-vp.pkl" \
DATASET=cifar10 \
NUM_SAMPLES=500 \
NODE_BUDGET=500 \
GPUS=0,1 \
EPS="0.1 0.3 0.6" \
MATCH=verification \
OUT_ROOT="$PWD/results/edm/cifar10/cifar10_n500_leaves_match-verification" \
RULES=paws \
bash experiments/images/sweep.sh

echo "All four sweeps completed."