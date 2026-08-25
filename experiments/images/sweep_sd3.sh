#!/usr/bin/env bash
# The (K, L) sweep on SD3.5: d-grs vs rmc, generation only. CLIP scoring is a
# separate step (the command is printed at the end).
#
#     nohup bash experiments/images/sweep_sd3.sh > /path/sd3_sweep.log 2>&1 &
#
# Resumable per cell, and per shard within a cell.
#
# COST. Wall clock tracks the *verification* budget |I|, doubled again by
# classifier-free guidance: a round pushes `sample_batch x |I| x 2` latents
# through the transformer. |I| is printed per cell below.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NETWORK="${NETWORK:-stabilityai/stable-diffusion-3.5-medium}"
PROMPTS="${PROMPTS:?set PROMPTS to a captions file, one per line}"

GPUS="${GPUS:-0,1,2,3}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_STEPS="${NUM_STEPS:-28}"
EPS="${EPS:-0.25}"
SEED="${SEED:-0}"
GUIDANCE="${GUIDANCE:-7.0}"
RESOLUTION="${RESOLUTION:-512}"
DTYPE="${DTYPE:-bfloat16}"
DECODE_BATCH="${DECODE_BATCH:-8}"
FORWARD_BATCH="${FORWARD_BATCH:-16}"
NEGATIVE="${NEGATIVE:-}"
# SD3.5-medium in bf16 is ~18 GiB of weights (T5-XXL dominates) before a single
# latent exists, so the bar for a usable GPU is much higher than for EDM.
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"

CONFIGS="${CONFIGS-2,2 3,2 2,3 3,3}"
RULES="${RULES-d-grs rmc}"
MATCH="${MATCH:-verification}"
INCLUDE_TARGET="${INCLUDE_TARGET:-1}"
# Latents per batched target call, BEFORE the CFG doubling. Divided by each
# cell's |I| to give that cell's --sample-batch, so memory stays flat as K grows.
NODE_BUDGET="${NODE_BUDGET:-64}"

OUT_ROOT="${OUT_ROOT:-$REPO/results/sd3/n${NUM_SAMPLES}_eps${EPS}_cfg${GUIDANCE}_${MATCH}}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

[[ -f "$PROMPTS" ]] || fail "prompts file not found: $PROMPTS"
avail="$(grep -c . "$PROMPTS")"
(( avail >= NUM_SAMPLES )) \
  || fail "$PROMPTS has $avail captions, NUM_SAMPLES=$NUM_SAMPLES needs one each"
mkdir -p "$OUT_ROOT" || fail "cannot write $OUT_ROOT"

NUM_PROC="$(awk -F, '{print NF}' <<< "$GPUS")"
SPEC_STEPS=$(( NUM_STEPS - 2 ))

busy=""
for g in ${GPUS//,/ }; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)" \
    || fail "cannot query GPU $g"
  log "gpu $g    : ${free} MiB free"
  (( free < MIN_FREE_MIB )) && busy="$busy $g"
done
[[ -n "$busy" ]] && fail "GPU(s)$busy have < ${MIN_FREE_MIB} MiB free. SD3.5-medium
       needs ~18 GiB of weights before any latents; drop them from GPUS=."

cd "$REPO"

log "output   : $OUT_ROOT"
log "network  : $NETWORK"
log "prompts  : $PROMPTS (first $NUM_SAMPLES of $avail, noise seed i = $SEED + i)"
log "sampling : $NUM_SAMPLES samples, T=$NUM_STEPS ($SPEC_STEPS speculative), eps=$EPS"
log "sd3      : cfg=$GUIDANCE  ${RESOLUTION}px  $DTYPE"
log "gpus     : $GPUS ($NUM_PROC processes)"
log "configs  : $CONFIGS   rules: $RULES   match: $MATCH"

verified_nodes() {   # $1=rule $2=K $3=L
  python - "$1" "$2" "$3" "$SPEC_STEPS" "$MATCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "experiments"))
from images.run_sd3 import matched_chain_depth          # noqa: E402
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

log "cost per cell (|I| latents per target call, doubled again by CFG):"
for cfg in $CONFIGS; do
  K="${cfg%%,*}"; L="${cfg##*,}"
  line="  (K=$K,L=$L)"
  for rn in $RULES; do line="$line  $rn |I|=$(verified_nodes "$rn" "$K" "$L")"; done
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
    experiments/images/run_sd3.py \
      --network "$NETWORK" --prompts "$PROMPTS" --negative-prompt "$NEGATIVE" \
      --rule "$rn" --branching "$K" --lookahead "$L" --match "$MATCH" \
      --num-samples "$NUM_SAMPLES" --num-steps "$NUM_STEPS" --eps "$EPS" \
      --seed "$SEED" --guidance-scale "$GUIDANCE" --resolution-px "$RESOLUTION" \
      --dtype "$DTYPE" --sample-batch "$sb" --forward-batch "$FORWARD_BATCH" \
      --decode-batch "$DECODE_BATCH" --out "$out" \
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

dirs=""
if [[ "$INCLUDE_TARGET" == "1" ]]; then
  run_cell "target" "$OUT_ROOT/plain-target" 1 1
  dirs=" $OUT_ROOT/plain-target"
fi
for cfg in $CONFIGS; do
  K="${cfg%%,*}"; L="${cfg##*,}"
  for rn in $RULES; do
    run_cell "$rn" "$OUT_ROOT/K${K}_L${L}/$rn" "$K" "$L"
    dirs="$dirs $OUT_ROOT/K${K}_L${L}/$rn"
  done
done

log "generation complete. Score prompt faithfulness with:"
echo
echo "  python experiments/images/clip.py --samples$dirs \\"
echo "      --device cuda:${GPUS%%,*} --output $OUT_ROOT/clip_report.json"
echo
log "no FID: these are samples of a text conditional, not of a dataset"
log "distribution. NFE and acceptance are in each meta.json; grids are the"
log "visual check that every rule samples the same law at temperature 1."
