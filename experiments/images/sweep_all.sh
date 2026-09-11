#!/usr/bin/env bash
# Both EDM sweeps, each at the NODE_BUDGET its resolution can actually hold.
#
#     nohup bash experiments/images/sweep_all.sh > results/edm/sweep_all.log 2>&1 &
#
# A thin wrapper over sweep.sh: it runs the same (K, L) x rule x eps grid twice,
# once per checkpoint, and its only real content is one number per dataset.
#
# WHY THE BUDGETS DIFFER. NODE_BUDGET caps the target rows in one batched
# forward; sweep.sh divides it by each cell's |I| to get that cell's
# --sample-batch, so peak memory stays flat across the grid. Activations go as
# rows x H^2, so the same card holds four times fewer 64x64 rows than 32x32
# ones: a budget that fits CIFAR-10 does not fit FFHQ. The values below are the
# ones each sweep has already run at on 2x10 GiB (peak 497 rows at 32x32, 98 at
# 64x64); 500 on FFHQ would be ~4x the largest footprint that has ever fit.
#
# Raising them buys almost nothing anyway. Total work is |I| x sum_i r_i,
# independent of how it is chunked, and throughput is already saturated: 890
# states/s at 497 rows vs 798 at 100 (CIFAR-10), and flat across 80-100 rows on
# FFHQ. NODE_BUDGET is a memory knob, not a speed one -- and it does move the
# reported batch speed-ups, since a larger --sample-batch takes a larger
# straggler hit. Use FORWARD_BATCH to cut memory without touching either.
#
# Resumable, because sweep.sh is: any cell with samples.pt and meta.json is
# skipped, so re-running continues where a crash or a Ctrl-C left off.
# DRY_RUN=1 prints the two invocations and exits.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# CIFAR-10 first: it is the cheaper half, so a mistake in the protocol surfaces
# in five hours rather than in nineteen.
DATASETS="${DATASETS:-cifar10 ffhq}"

CIFAR10_NETWORK="${CIFAR10_NETWORK:-$REPO/edm/edm-cifar10-32x32-cond-vp.pkl}"
FFHQ_NETWORK="${FFHQ_NETWORK:-$REPO/edm/edm-ffhq-64x64-uncond-vp.pkl}"
CIFAR10_NODE_BUDGET="${CIFAR10_NODE_BUDGET:-500}"      # peak 497 rows at 32x32
FFHQ_NODE_BUDGET="${FFHQ_NODE_BUDGET:-100}"            # peak  98 rows at 64x64

# Everything below is passed through to sweep.sh unchanged, so its own defaults
# stay the single source of truth for the protocol.
GPUS="${GPUS:-0,1}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
export EPS="${EPS:-0.1 0.3 0.6}"
export NUM_STEPS="${NUM_STEPS:-100}"
export SEED="${SEED:-0}"
export MATCH="${MATCH:-verification}"
export CONFIGS="${CONFIGS-2,3 2,5 3,4 4,4}"
export RULES="${RULES-d-grs rmc paws}"
export FORWARD_BATCH="${FORWARD_BATCH:-0}"
export S_NOISE="${S_NOISE:-1.0}"
export INCLUDE_TARGET="${INCLUDE_TARGET:-1}"
DATA="${DATA:-}"                    # ffhq: real images, for the FID it prints

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

network_for() { case "$1" in cifar10) echo "$CIFAR10_NETWORK";;
                             ffhq)    echo "$FFHQ_NETWORK";;
                             *) fail "no network for dataset '$1' (set ${1^^}_NETWORK)";; esac; }
budget_for()  { case "$1" in cifar10) echo "$CIFAR10_NODE_BUDGET";;
                             ffhq)    echo "$FFHQ_NODE_BUDGET";;
                             *) fail "no node budget for dataset '$1'";; esac; }

# Both checkpoints up front: finding the second one missing after the first
# sweep has run is a wasted afternoon, and sweep.sh only checks its own.
for ds in $DATASETS; do
  net="$(network_for "$ds")"
  [[ -f "$net" ]] || fail "$ds: network not found: $net"
done

log "repo     : $REPO"
log "datasets : $DATASETS   ($NUM_SAMPLES samples each, eps: $EPS)"
for ds in $DATASETS; do
  log "  $ds: NODE_BUDGET=$(budget_for "$ds")  $(network_for "$ds")"
done
log "estimate : ~5 h cifar10 + ~14 h ffhq on 2 GPUs, scaled from the n=100 sweep"

for ds in $DATASETS; do
  net="$(network_for "$ds")"; nb="$(budget_for "$ds")"
  cmd=(env NETWORK="$net" DATASET="$ds" NUM_SAMPLES="$NUM_SAMPLES"
       NODE_BUDGET="$nb" GPUS="$GPUS" ${DATA:+DATA="$DATA"}
       bash "$REPO/experiments/images/sweep.sh")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %q ' "${cmd[@]}"; printf '\n'
    continue
  fi
  mkdir -p "$REPO/results/edm"
  sweep_log="$REPO/results/edm/sweep_${ds}_n${NUM_SAMPLES}.log"
  log "=== $ds: starting (NODE_BUDGET=$nb), tee -> $sweep_log"
  "${cmd[@]}" 2>&1 | tee "$sweep_log" \
    || fail "$ds sweep failed -- see $sweep_log. Fix the cause and re-run this
       script: finished cells are skipped, so it resumes at the broken one."
  log "=== $ds: done"
done

log "all sweeps complete. Figures:"
echo
echo "  python experiments/images/plot_edm.py --root results/edm \\"
echo "      --out results/edm/figures/speedup_vs_budget"
echo
log "per-image NFE counts are now in each meta.json, so the isolated metric"
log "carries its own error band:"
echo
echo "  python experiments/images/plot_edm.py --root results/edm \\"
echo "      --metric mean_isolated_speedup --over images \\"
echo "      --out results/edm/figures/speedup_isolated"
echo
