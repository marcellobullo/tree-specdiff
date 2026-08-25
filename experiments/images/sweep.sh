#!/usr/bin/env bash
# The (K, L) sweep on pretrained EDM: d-grs vs rmc, generation only.
# FID is computed afterwards from the saved samples (command printed at the end).
#
#     nohup bash experiments/images/sweep.sh > /path/sweep.log 2>&1 &
#
# Resumable: any (K, L, rule) cell whose samples.pt and meta.json both exist is
# skipped, so a crash, an OOM from a neighbouring job, or a deliberate stop
# costs only the unfinished cell. Re-run the same command to continue. Within a
# cell, per-rank shards are reused too, so a cell that died halfway through
# restarts at the shard boundary rather than at zero.
#
# COST. Wall clock scales with the *verification* budget |I| -- the states the
# expensive network evaluates per round -- not with the NFE the method reports
# and not with the proposal budget B. |I| is printed per cell below; it is also
# what has to fit in memory, and what SAMPLE_BATCH is derived from.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NETWORK="${NETWORK:?set NETWORK to a pretrained EDM .pkl}"
EDM_REPO="${EDM_REPO:?set EDM_REPO to a NVlabs/edm checkout}"
DATASET="${DATASET:-cifar10}"          # cifar10 | ffhq  -- scoring only
DATA="${DATA:-}"                       # ffhq: image dir or zip for the real set

GPUS="${GPUS:-0,1,2,3}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
NUM_STEPS="${NUM_STEPS:-100}"
EPS="${EPS:-0.25}"
SEED="${SEED:-0}"
LABELS="${LABELS:-auto}"               # auto | uniform | none | <class index>
FORWARD_BATCH="${FORWARD_BATCH:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"

# "K,L" pairs, in the order they run. `-` not `:-`, so CONFIGS="" means NO
# configs (INCLUDE_TARGET=1 CONFIGS="" generates the baseline alone); with `:-`
# an empty value would silently restore the whole default sweep.
CONFIGS="${CONFIGS-2,2 3,2 4,2 5,2 2,3 3,3 4,3 2,4 3,4}"
RULES="${RULES-d-grs rmc}"
# How the rmc chain is sized against each (K, L) tree. `verification` gives both
# arms the same target batch |I| -- the hardware-matched comparison, and
# gm_sweep.py's default. `budget` is the paper's protocol, chain(B), a factor of
# K longer. The choice is recorded in every meta.json.
MATCH="${MATCH:-verification}"
# A plain-target run at the same eps: the matched-speed baseline, and the
# reference FID the speculative rules must match (they sample the same law).
INCLUDE_TARGET="${INCLUDE_TARGET:-1}"
# Verified nodes per batched target call. Divided by each cell's |I| to get that
# cell's --sample-batch, so memory stays roughly flat across the grid rather
# than growing with K.
NODE_BUDGET="${NODE_BUDGET:-2000}"

OUT_ROOT="${OUT_ROOT:-$REPO/results/edm/${DATASET}_n${NUM_SAMPLES}_eps${EPS}_${MATCH}}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

[[ -d "$REPO" ]]     || fail "repo not found: $REPO"
[[ -f "$NETWORK" ]]  || fail "network not found: $NETWORK"
[[ -d "$EDM_REPO" ]] || fail "edm checkout not found: $EDM_REPO"
mkdir -p "$OUT_ROOT" || fail "cannot write $OUT_ROOT"

NUM_PROC="$(awk -F, '{print NF}' <<< "$GPUS")"
SPEC_STEPS=$(( NUM_STEPS - 2 ))        # the two Euler endpoints are not speculative

# Refuse to launch onto a GPU a neighbouring job already owns. One rank OOMing
# in build_denoiser is not a local failure: the ranks that DID fit generate
# their whole shard and then block on the post-generation barrier waiting for a
# peer that has already died. Checking here is the difference between a clear
# message now and a silent stall hours later.
busy=""
for g in ${GPUS//,/ }; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)" \
    || fail "cannot query GPU $g (is it a valid index?)"
  log "gpu $g    : ${free} MiB free"
  (( free < MIN_FREE_MIB )) && busy="$busy $g"
done
[[ -n "$busy" ]] && fail "GPU(s)$busy have < ${MIN_FREE_MIB} MiB free. Drop them from
       GPUS=, wait for the other job, or lower MIN_FREE_MIB= if you know it fits."

cd "$REPO"

log "output   : $OUT_ROOT"
log "network  : $NETWORK"
log "sampling : $NUM_SAMPLES samples, T=$NUM_STEPS ($SPEC_STEPS speculative), eps=$EPS"
log "labels   : $LABELS"
log "gpus     : $GPUS ($NUM_PROC processes)"
log "configs  : $CONFIGS   rules: $RULES   match: $MATCH"

# |I| per cell, from the library rather than a formula duplicated here -- the
# two must not be able to disagree.
verified_nodes() {   # $1=rule $2=K $3=L
  python - "$1" "$2" "$3" "$SPEC_STEPS" "$MATCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "experiments"))
from images.run_edm import matched_chain_depth          # noqa: E402
from specdiff import DraftTree                          # noqa: E402

rule, K, L, steps, match = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if rule == "target":
    print(1)
else:
    tree = DraftTree.uniform(branching=K, lookahead=L)
    if rule == "rmc":
        tree = DraftTree.chain(matched_chain_depth(tree, steps, match))
    print(tree.verification_budget())
PY
}

log "cost per cell (|I| = target rows per round, the quantity wall clock tracks):"
for cfg in $CONFIGS; do
  K="${cfg%%,*}"; L="${cfg##*,}"
  line="  (K=$K,L=$L)"
  for rn in $RULES; do
    line="$line  $rn |I|=$(verified_nodes "$rn" "$K" "$L")"
  done
  echo "$line"
done

run_cell() {   # $1=rule  $2=out  $3=K  $4=L
  local rn="$1" out="$2" K="$3" L="$4"
  if [[ -f "$out/samples.pt" && -f "$out/meta.json" ]]; then
    log "skip $rn K=$K L=$L (already done)"; return 0
  fi
  local iv sb
  iv="$(verified_nodes "$rn" "$K" "$L")"
  sb=$(( NODE_BUDGET / iv )); (( sb < 1 )) && sb=1
  (( sb > NUM_SAMPLES )) && sb=$NUM_SAMPLES
  mkdir -p "$out"
  log "generating $rn K=$K L=$L  (|I|=$iv, sample-batch $sb) -> $out"
  local mp=(); (( NUM_PROC > 1 )) && mp=(--multi_gpu)
  accelerate launch "${mp[@]}" --num_processes "$NUM_PROC" --gpu_ids "$GPUS" \
    experiments/images/run_edm.py \
      --network "$NETWORK" --edm-repo "$EDM_REPO" \
      --rule "$rn" --branching "$K" --lookahead "$L" --match "$MATCH" \
      --num-samples "$NUM_SAMPLES" --num-steps "$NUM_STEPS" --eps "$EPS" \
      --seed "$SEED" --labels "$LABELS" \
      --sample-batch "$sb" --forward-batch "$FORWARD_BATCH" \
      --out "$out" \
    > "$out/generate.log" 2>&1 \
    || fail "failed: $rn K=$K L=$L -- see $out/generate.log"
  log "done $rn K=$K L=$L: $(python - "$out/meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"speedup {m['speedup']:.2f}x (end-to-end {m['end_to_end_speedup']:.2f}x)  "
      f"acc {m['acceptance_rate']:.3f}  occupancy {m['occupancy']:.2f}  {m['seconds']:.0f}s")
PY
)"
}

# Baseline first: cheapest cell, and the reference FID every other cell must
# match at temperature 1.
dirs=""
if [[ "$INCLUDE_TARGET" == "1" ]]; then
  run_cell "target" "$OUT_ROOT/plain-target" 1 1
  dirs=" $OUT_ROOT/plain-target"          # leading space: `--samples$dirs`
fi

for cfg in $CONFIGS; do
  K="${cfg%%,*}"; L="${cfg##*,}"
  for rn in $RULES; do
    run_cell "$rn" "$OUT_ROOT/K${K}_L${L}/$rn" "$K" "$L"
    dirs="$dirs $OUT_ROOT/K${K}_L${L}/$rn"
  done
done

log "generation complete. NFE / acceptance are in:"
echo
for d in $dirs; do echo "  $d/meta.json"; done
echo
# Scoring is not ported into specdiff yet -- the reference implementation's
# scorer needs its Inception statistics and dataset loaders, which are a
# separate piece of work. samples.pt is written in exactly the layout that
# scorer expects (uint8 (N, C, H, W)), so it can be used unchanged -- and using
# it keeps these FIDs directly comparable with the ones already computed there.
if [[ -n "${FID_REPO:-}" ]]; then
  log "score with the reference implementation:"
  echo
  echo "  cd $FID_REPO && python scripts/fid_from_samples.py --samples$dirs \\"
  echo "      --cache-dir $OUT_ROOT/_fid_cache --device cuda:${GPUS%%,*} \\"
  echo "      --dataset $DATASET${DATA:+ --data $DATA} \\"
  echo "      --num-real 50000 --inception-score --output $OUT_ROOT/fid_report.json"
  echo
  if [[ "$DATASET" != "cifar10" && -z "$DATA" ]]; then
    log "note: --dataset $DATASET needs --data (the real images); set DATA= to fill it in above"
  fi
else
  log "set FID_REPO=<accelerating-diffusion-sampling checkout> to have the"
  log "scoring command printed here; specdiff has no scorer of its own yet."
fi
