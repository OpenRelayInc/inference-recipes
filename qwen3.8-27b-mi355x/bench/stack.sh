#!/usr/bin/env bash
# stack.sh <name> <gpu> <port> [feature ...] [-- extra engine args]: launches v0.30.0 + BASE_ARGS (production flags)
# with the named recipe pieces added, so every ablation stage is one command line.
#   gemm      the 75 power-of-two tuned a8w8 block-scale B-preshuffle GEMM rows (bench/stages/gemm_pow2.csv)
#   gemm2     gemm plus 28 exact-M decode/MTP-verify rows: the shipped table (qwen38_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)
#   ua1       unified-attention table with the D_GEQ_256.Q_GEQ_256.DT_fp8_fp8 prefill entry (unified_attention_gfx950.json)
#   ua=<file> another unified-attention table (path relative to the recipe dir)
#   ssm16     --mamba-ssm-cache-dtype bfloat16
#   gdn       patch_gdn_aiter_decode.py (aiter fused GDN decode for the flat qkvz layout)
#   mtp<k>    --speculative-config {"method":"mtp","num_speculative_tokens":k}
#   mbt<n>    --max-num-batched-tokens n
#   image=<i> engine image (default vllm/vllm-openai-rocm:v0.30.0)
source "$(dirname "$0")/common.sh"; source "$(dirname "$0")/engine_args.sh"
name=$1 gpu=$2 port=$3; shift 3
img=vllm/vllm-openai-rocm:v0.30.0
AC=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs
UAD=/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json
mounts=() pre=() args=("${BASE_ARGS[@]}")
while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in
    gemm) mounts+=("$RECIPE_DIR/bench/stages/gemm_pow2.csv:$AC/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen38_27b.csv:ro") ;;
    gemm2) mounts+=("$RECIPE_DIR/qwen38_a8w8_blockscale_bpreshuffle_tuned_gemm.csv:$AC/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen38_27b.csv:ro") ;;
    ua1) mounts+=("$RECIPE_DIR/unified_attention_gfx950.json:$UAD:ro") ;;
    ua=*) mounts+=("$RECIPE_DIR/${1#ua=}:$UAD:ro") ;;
    ssm16) args+=(--mamba-ssm-cache-dtype bfloat16) ;;
    gdn) pre+=("python3 /recipe/patch_gdn_aiter_decode.py") ;;
    mtp*) args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${1#mtp}}") ;;
    mbt*) args+=(--max-num-batched-tokens "${1#mbt}") ;;
    image=*) img=${1#image=} ;;
    *) echo "unknown feature $1" >&2; exit 1 ;;
  esac; shift
done
[[ ${1:-} == -- ]] && shift
PRE=$(IFS='&'; echo "${pre[*]:-true}" | sed 's/&/ \&\& /g') EXTRA_MOUNTS="${mounts[*]:-}" \
  "$BENCH_DIR/launch.sh" "$name" "$gpu" "$port" "$img" "${args[@]}" "$@"
