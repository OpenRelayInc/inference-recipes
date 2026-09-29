# vLLM ROCm engine tuned for Gemma 4 31B (AMD Quark MXFP4) on AMD MI355X (gfx950).
# What each layer does and the measurements behind it: README.md in this directory.
FROM vllm/vllm-openai-rocm:v0.30.0

ARG SITE=/usr/local/lib/python3.12/dist-packages

# aiter's gfx950 unified-attention table plus prefill tiles for head_dim 512 and 256 sliding-window,
# and split-KV settings for head_dim 512 decode. The shipped table ran both Gemma 4 prefill paths on a
# decode-sized tile.
COPY unified_attention_gfx950.json ${SITE}/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json

# Tuned MXFP4 (a4w4) GEMM rows for Gemma 4 31B's shapes. aiter merges every CSV under model_configs/.
COPY gemma4_a4w4_blockscale_tuned_gemm.csv ${SITE}/aiter/configs/model_configs/gemma4_a4w4_blockscale_tuned_gemm.csv

# Source edits, applied at build time. Both fail the build if their anchor is missing.
#   patch_g4_fuse.py: fused GELU(tanh)*up + MXFP4 quant into the down_proj GEMM (active only with
#   G4_FUSE_GELU=1), and aiter's scaleM_pad made a runtime argument so the fused kernel does not
#   JIT-compile a new variant for every batch size mid-serving.
#   patch_aiter_ua3d.py: split-KV 3D attention for head_dim >= 512 on short-query (decode / MTP verify) steps.
COPY patch_g4_fuse.py patch_aiter_ua3d.py /opt/openrelay/
RUN python3 /opt/openrelay/patch_g4_fuse.py && python3 /opt/openrelay/patch_aiter_ua3d.py \
 && grep -q "g4_gelu_mxfp4_gemm" ${SITE}/vllm/model_executor/models/gemma4.py \
 && grep -q "params.max_seqlen_q > 16" ${SITE}/aiter/ops/triton/attention/unified_attention.py

# Keep Triton's JIT cache under /root/.cache. Mount a persistent volume there so a restarted engine
# does not recompile the attention and fusion kernels while serving.
ENV G4_FUSE_GELU=1 \
    TRITON_CACHE_DIR=/root/.cache/triton
