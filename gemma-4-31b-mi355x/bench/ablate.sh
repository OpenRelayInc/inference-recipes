#!/usr/bin/env bash
# ablate.sh [gpu] [port]: cumulative ablation on ONE GPU, so GPU-to-GPU variance cannot confound the per-stage
# gains. Each stage adds one change to the stage before it. For each stage: launch, smoke, 3 prefill probes
# (10K in, c=16, 64 requests), 1 closed-loop probe at the workload's shape (c=32, 9.5K in / 380 out, 128 requests).
# Produced results/ablation.md. IMAGE must be the stock image (the default); every tuned piece is mounted or
# patched in per stage. The shipped config is stage G plus --max-num-batched-tokens 8192.
# gpu and port default to $GPU (0) and $PORT (8007); the engine container is $NAME (g4-ablate).
source "$(dirname "$0")/common.sh"; cd "$BENCH_DIR" || exit 1; source ./engine_args.sh
gpu=${1:-${GPU:-0}} port=${2:-${PORT:-8007}} name=${NAME:-g4-ablate}
MC=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/gemma4_a4w4_blockscale_tuned_gemm.csv
CSV="$RECIPE_DIR/gemma4_a4w4_blockscale_tuned_gemm.csv:$MC:ro"; KC="{\"ir_op_priority\":{\"rms_norm\":[\"native\"]}}"
P2="python3 /recipe/patch_g4_fuse.py"; P3="python3 /recipe/patch_g4_fuse.py && python3 /recipe/patch_aiter_ua3d.py"
UA1=$BENCH_DIR/stages/ua_v1.json UA2=$BENCH_DIR/stages/ua_v2.json UA3=$RECIPE_DIR/unified_attention_gfx950.json
run() { # stage name, then engine args; per-stage settings come in as env assignments on the call
  local stage=$1; shift
  ./launch.sh "$name" "$gpu" "$port" "$IMAGE" "$WEIGHTS" "$@" >/dev/null || { echo "$stage launch failed"; return; }
  for _ in $(seq 120); do curl -sf "localhost:$port/v1/models" >/dev/null && break; sleep 5; done
  echo "== $stage: $(./smoke.sh "$port")"
  for _ in 1 2 3; do ./probe.sh "$port" 16 10000 1 64 | grep "Request thr"; done
  ./probe.sh "$port" 32 9500 380 128 | grep -E "Request thr|Mean TPOT|Mean TTFT"
}
docker rm -f "$name" >/dev/null 2>&1
run A_base "${BASE_ARGS[@]}"
UA_JSON=$UA1 run B_ua1 "${BASE_ARGS[@]}"
UA_JSON=$UA1 EXTRA_MOUNTS=$CSV run C_gemm "${BASE_ARGS[@]}"
UA_JSON=$UA2 EXTRA_MOUNTS=$CSV run D_cg "${BASE2_ARGS[@]}"
UA_JSON=$UA2 EXTRA_MOUNTS=$CSV PRE="$P2" EXTRA_ENV="G4_FUSE_GELU=1" run E_fuse "${BASE2_ARGS[@]}"
UA_JSON=$UA3 EXTRA_MOUNTS=$CSV PRE="$P3" EXTRA_ENV="G4_FUSE_GELU=1" run F_3d "${BASE2_ARGS[@]}"
UA_JSON=$UA3 EXTRA_MOUNTS=$CSV PRE="$P3" EXTRA_ENV="G4_FUSE_GELU=1" run G_native "${BASE2_ARGS[@]}" --kernel-config "$KC"
docker rm -f "$name" >/dev/null 2>&1
