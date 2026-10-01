# Qwen3.8-27B-FP8 on AMD MI355X (gfx950): stock vLLM v0.30.0 ROCm image plus two tuned config tables.
#   docker build -t vllm-rocm-qwen38:v0.30.0-q38mi355 .
# No GPU needed to build.
FROM vllm/vllm-openai-rocm:v0.30.0

ARG SITE=/usr/local/lib/python3.12/dist-packages

# Tuned aiter a8w8 block-scale (B-preshuffle) GEMM rows for the model's five FP8 linear shapes. aiter merges every
# *a8w8_blockscale_bpreshuffle_tuned_gemm*.csv under model_configs/ at start; with rows present for (N, K), vLLM
# selects AiterPreshuffledFp8BlockScaledMMKernel instead of the CK default instance.
COPY qwen38_a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
     ${SITE}/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen38_27b.csv

# aiter's gfx950 unified-attention table with a prefill entry for head_dim 256 + fp8 q/KV (D_GEQ_256.Q_GEQ_256.DT_fp8_fp8).
COPY unified_attention_gfx950.json \
     ${SITE}/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json

# patch_gdn_aiter_decode.py is not applied: with MTP every decode step carries speculative tokens and vLLM never
# takes the patched path (no aiter GDN kernels in the shipped config's trace). Apply it only for an MTP-off pool:
#   COPY patch_gdn_aiter_decode.py /tmp/ && RUN python3 /tmp/patch_gdn_aiter_decode.py

ENV VLLM_ROCM_USE_AITER=1 \
    TRITON_CACHE_DIR=/root/.cache/triton
