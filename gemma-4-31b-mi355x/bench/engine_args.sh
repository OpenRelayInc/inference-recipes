# shellcheck shell=bash disable=SC2034
# engine_args.sh: the engine arguments used in the run. Sourced, not executed; then pass
# "${BASE_ARGS[@]}" (or another array) to launch.sh, which adds
# --max-model-len 32768 --max-num-seqs 64 --enable-prompt-tokens-details.

# The first working MI355X config, and the baseline for every comparison in results/.
BASE_ARGS=(--enable-prefix-caching
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
  --speculative-config '{"method":"mtp","model":"/weights/assistant","num_speculative_tokens":2}'
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 --max-num-batched-tokens 16384
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,64]}')

# EXP-3: CUDA-graph capture sizes that cover MTP decode batches (64 seqs x 3 tokens = 192 tokens).
BASE2_ARGS=("${BASE_ARGS[@]:0:${#BASE_ARGS[@]}-2}" --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,224,256]}')

# EXP-6 (discarded): capture sizes for up to 128 seqs x 3 tokens, used with MAXSEQS=128.
BASE3_ARGS=("${BASE_ARGS[@]:0:${#BASE_ARGS[@]}-2}" --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,224,256,288,320,352,384]}')

# The shipped config ("b8k3"): BASE2_ARGS with native RMSNorm and an 8192-token prefill budget.
# Needs the tuned image, or the stock image with UA_JSON, the GEMM CSV mount, both patches in PRE and
# EXTRA_ENV=G4_FUSE_GELU=1 (ablate.sh stage G shows the stock-image form).
TUNED_ARGS=(--enable-prefix-caching
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
  --speculative-config '{"method":"mtp","model":"/weights/assistant","num_speculative_tokens":2}'
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 --max-num-batched-tokens 8192
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,224,256]}'
  --kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}')
