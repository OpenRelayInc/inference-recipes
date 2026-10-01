# shellcheck shell=bash disable=SC2034
# engine_args.sh: engine argument arrays. Sourced, not executed.
# The production deployment's flags (the baseline to beat), minus --model/--served-model-name which launch.sh adds.
PROD_ARGS=(--tensor-parallel-size 1 --data-parallel-size 1 --max-model-len 262144 --max-num-seqs 64
  --enable-prefix-caching --enable-prompt-tokens-details --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_xml)
BASE_ARGS=("${PROD_ARGS[@]}" --kv-cache-dtype fp8 --attention-backend ROCM_AITER_UNIFIED_ATTN)
# Baseline C: engine defaults for attention backend and KV dtype.
DEFAULT_ARGS=("${PROD_ARGS[@]}")
