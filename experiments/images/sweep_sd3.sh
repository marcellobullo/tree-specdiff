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

GPUS="${GPUS:-0,1}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_STEPS="${NUM_STEPS:-50}"
EPS="${EPS:-0.8}"
SEED="${SEED:-0}"
GUIDANCE="${GUIDANCE:-7.0}"
RESOLUTION="${RESOLUTION:-512}"
DTYPE="${DTYPE:-bfloat16}"
DECODE_BATCH="${DECODE_BATCH:-8}"
FORWARD_BATCH="${FORWARD_BATCH:-16}"
NEGATIVE="${NEGATIVE:-}"
# Where the text encoders run. Empty keeps them with the transformer, which
# needs all ~18 GiB resident at once. Set ENCODE_DEVICE=cpu below ~16 GiB VRAM:
# only the transformer and the VAE then reach the GPU (~4.8 GiB), at the cost of
# a few minutes of CPU encoding per cell.
ENCODE_DEVICE="${ENCODE_DEVICE:-}"
# Cache of the encoded captions. Every cell of a grid encodes the same caption
# set with the same model and the same negative prompt, so one encode can serve
# all of them instead of one per cell. Shared across sweeps on purpose: the key
# covers what changes the numbers, and eps / cfg / match are not among them.
# `-` not `:-`, so CACHE_ENCODED_PROMPTS= disables it. About 270 MB per 100
# captions, so check the free space before pointing it at a 30k set.
CACHE_ENCODED_PROMPTS="${CACHE_ENCODED_PROMPTS-$REPO/results/sd3/_prompt_cache}"
# SD3.5-medium in bf16 is ~18 GiB of weights (T5-XXL dominates) before a single
# latent exists, so the bar for a usable GPU is much higher than for EDM.
#MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"

CONFIGS="${CONFIGS-2,2 3,2 4,2 5,2 6,2 2,3 3,3 4,3 5,3 6,3}"
#CONFIGS="${CONFIGS-2,3}"
RULES="${RULES-d-grs paws rmc}"
MATCH="${MATCH:-verification}"
# Sampler options, S_noise and the timestep shift: protocol that does not appear
# in a cell's output path, so the guard in run_cell is the only thing stopping
# two settings being pooled into one grid. Same treatment as sweep.sh.
SAMPLER_CONFIG="${SAMPLER_CONFIG:-}"
# Per-rule JSON options, included in resume checks and passed to every cell.
VERIFIER_OPTIONS="${VERIFIER_OPTIONS:-}"
[[ -n "$VERIFIER_OPTIONS" ]] || VERIFIER_OPTIONS='{}'
S_NOISE="${S_NOISE:-1.0}"
# SD3.5 is trained at 1024px and its noise schedule is resolution dependent, so
# 512px generation shifts the grid by t -> kt / (1 + (k-1)t). 3.0 is what the
# pipeline ships and what run_sd3.py defaults to; it is passed explicitly here
# so a change to that default cannot move a grid without the guard noticing.
SHIFT="${SHIFT:-3.0}"
INCLUDE_TARGET="${INCLUDE_TARGET:-1}"
# Latents per batched target call, BEFORE the CFG doubling. Divided by each
# cell's |I| to give that cell's --sample-batch, so memory stays flat as K grows.
NODE_BUDGET="${NODE_BUDGET:-64}"
# Fixes --sample-batch instead of deriving it from |I|. Leave empty for the
# NODE_BUDGET behaviour above. Set it when comparing two sweeps that differ in
# |I| -- evaluate_leaves is the case that matters: a derived batch makes the
# batch size a function of the treatment, and trajectories sharing an RNG
# stream are not reproducible across batch sizes (specdiff/batched.py), so the
# two grids would differ in their randomness as well as in the thing under
# test. Memory is then yours to check: the peak scales with SAMPLE_BATCH x |I|,
# so size it from the largest |I| in CONFIGS.
SAMPLE_BATCH="${SAMPLE_BATCH:-}"

OUT_ROOT="${OUT_ROOT:-$REPO/results/sd3/n${NUM_SAMPLES}_eps${EPS}_cfg${GUIDANCE}_${MATCH}}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

[[ -f "$PROMPTS" ]] || fail "prompts file not found: $PROMPTS"
avail="$(grep -c . "$PROMPTS")"
(( avail >= NUM_SAMPLES )) \
  || fail "$PROMPTS has $avail captions, NUM_SAMPLES=$NUM_SAMPLES needs one each"
mkdir -p "$OUT_ROOT" || fail "cannot write $OUT_ROOT"

NUM_PROC="$(awk -F, '{print NF}' <<< "$GPUS")"
SPEC_STEPS=$NUM_STEPS

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
log "sampling : $NUM_SAMPLES samples, T=$NUM_STEPS ($SPEC_STEPS sampler transitions), eps=$EPS"
log "sd3      : cfg=$GUIDANCE  ${RESOLUTION}px  $DTYPE  shift=$SHIFT"
log "encode   : ${ENCODE_DEVICE:-with the transformer}  cache ${CACHE_ENCODED_PROMPTS:-off}"
log "gpus     : $GPUS ($NUM_PROC processes)"
log "configs  : $CONFIGS   rules: $RULES   match: $MATCH"
POLICY_RESOLVED="$(python - "$SAMPLER_CONFIG" "$S_NOISE" "$SHIFT" "$VERIFIER_OPTIONS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "experiments"))
from images.run_common import load_sampler_config          # noqa: E402
from experiments.verifier_config import parse_verifier_options

print(json.dumps({"endpoint_policy": "in_sampler",
                  "nfe_accounting": "logical_calls_per_image_v2",
                  "verifier_options": parse_verifier_options(sys.argv[4]),
                  "sampler": load_sampler_config(sys.argv[1] or None),
                  "s_noise": float(sys.argv[2]),
                  "shift": float(sys.argv[3])}, sort_keys=True))
PY
)" || fail "could not resolve the sampler config"
log "policy   : $POLICY_RESOLVED"

verified_nodes() {   # $1=rule $2=K $3=L
  python - "$1" "$2" "$3" "$SPEC_STEPS" "$MATCH" "$SAMPLER_CONFIG" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "experiments"))
from images.run_common import load_sampler_config      # noqa: E402
from specdiff import DraftTree, create_verifier         # noqa: E402

rule, K, L, steps, match = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
leaves = load_sampler_config(sys.argv[6] or None)["evaluate_leaves"]
if rule == "target":
    tree = DraftTree.chain(1)
else:
    tree = DraftTree.uniform(branching=K, lookahead=L)
    tree = create_verifier(rule).matched_tree(
        tree, num_steps=steps, match=match, evaluate_leaves=leaves)
print(tree.verification_budget(evaluate_leaves=leaves))
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
    # Skipping on file existence says nothing about the settings that produced
    # the cell, and two cells of one grid at different settings are not
    # comparable -- nothing downstream would show it. So check.
    local was
    was="$(python - "$out/meta.json" <<'PY'
import json, sys

meta = json.load(open(sys.argv[1]))
# A cell with no `sampler` block predates the option, and the batched sampler
# had exactly one behaviour then; s_noise was likewise fixed at 1.0 and the
# shift at the pipeline's 3.0. Naming all three lets an older grid be continued
# deliberately rather than by accident.
print(json.dumps({
    "endpoint_policy": meta.get("endpoint_policy", "split"),
    "nfe_accounting": meta.get("nfe_accounting", "rounds_v1"),
    "sampler": meta.get("sampler", {"evaluate_leaves": False,
                                    "prefetch": "parent"}),
    "s_noise": meta.get("s_noise", 1.0),
    "verifier_options": meta.get("verifier_options", {}),
    "shift": meta.get("shift", 3.0),
}, sort_keys=True))
PY
)" || fail "cannot read $out/meta.json"
    if [[ "$was" != "$POLICY_RESOLVED" ]]; then
      fail "$out was generated under a different protocol:
         it has  : $was
         this run: $POLICY_RESOLVED
       Either delete the cell to regenerate it under this run's settings, or
       use a new OUT_ROOT for a different endpoint/accounting policy, or match the saved settings."
    fi
    log "skip $rn K=$K L=$L (already done)"; return 0
  fi
  local iv sb
  iv="$(verified_nodes "$rn" "$K" "$L")"
  if [[ -n "$SAMPLE_BATCH" ]]; then
    sb=$SAMPLE_BATCH
  else
    sb=$(( NODE_BUDGET / iv ))
  fi
  (( sb < 1 )) && sb=1
  (( sb > NUM_SAMPLES )) && sb=$NUM_SAMPLES
  mkdir -p "$out"
  log "generating $rn K=$K L=$L  (|I|=$iv, sample-batch $sb\
${SAMPLE_BATCH:+ fixed}) -> $out"
  local mp=(); (( NUM_PROC > 1 )) && mp=(--multi_gpu)
  local sc=(); [[ -n "$SAMPLER_CONFIG" ]] && sc=(--sampler-config "$SAMPLER_CONFIG")
  local ed=(); [[ -n "$ENCODE_DEVICE" ]] && ed=(--encode-device "$ENCODE_DEVICE")
  local pc=(); [[ -n "$CACHE_ENCODED_PROMPTS" ]] \
    && pc=(--prompt-cache "$CACHE_ENCODED_PROMPTS")
  accelerate launch "${mp[@]}" --num_processes "$NUM_PROC" --gpu_ids "$GPUS" \
    experiments/images/run_sd3.py \
      --network "$NETWORK" --prompts "$PROMPTS" --negative-prompt "$NEGATIVE" \
      --rule "$rn" --branching "$K" --lookahead "$L" --match "$MATCH" \
      --verifier-options "$VERIFIER_OPTIONS" \
      --num-samples "$NUM_SAMPLES" --num-steps "$NUM_STEPS" --eps "$EPS" \
      --seed "$SEED" --guidance-scale "$GUIDANCE" --resolution-px "$RESOLUTION" \
      --dtype "$DTYPE" --sample-batch "$sb" --forward-batch "$FORWARD_BATCH" \
      --decode-batch "$DECODE_BATCH" --s-noise "$S_NOISE" --shift "$SHIFT" \
      "${sc[@]}" "${ed[@]}" "${pc[@]}" --out "$out" \
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
