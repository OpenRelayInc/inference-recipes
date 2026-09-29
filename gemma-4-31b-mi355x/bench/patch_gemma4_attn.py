# Day-one experiment, kept for the record; not part of the recipe. With GEMMA4_ATTN_MODE=split the engine
# crashed under batching (HTTP 500s). ROCM_AITER_UNIFIED_ATTN on every layer (no patch) replaced it.
# To rerun: launch.sh with the stock image, EXTRA_ENV=GEMMA4_ATTN_MODE=split,
# PRE="python3 /recipe/bench/patch_gemma4_attn.py", and no --attention-backend flag.
"""Let vLLM pick the attention backend per layer for Gemma 4 on ROCm.

vLLM's Gemma4Config forces TRITON_ATTN on every layer when FA4 is missing (always
on ROCm). The Triton unified kernel is ~3x slower than AITER CK FA on the d=256
sliding layers. With GEMMA4_ATTN_MODE:
  auto  - drop the forcing; ROCm's normal per-layer priority applies
  split - drop the forcing and order ROCm priorities ROCM_AITER_FA first for
          head_size <= 256, ROCM_AITER_UNIFIED_ATTN for larger heads
"""
import os, re, sys
import vllm

root = os.path.dirname(vllm.__file__)
mode = os.environ.get("GEMMA4_ATTN_MODE", "auto")

cfg = os.path.join(root, "model_executor/models/config.py")
src = open(cfg).read()
old = "        elif vllm_config.attention_config.backend is None:\n            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN"
assert old in src, "Gemma4Config forcing block not found"
src = src.replace(old, "        elif False:\n            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN")
open(cfg, "w").write(src)

if mode == "split":
    fa = os.path.join(root, "v1/attention/backends/rocm_aiter_fa.py")
    s = open(fa).read()
    if "def supports_head_size" not in s:
        # CK FMHA supports head_dim <= 256; declare it so the selector skips it for 512.
        s = re.sub(r"(class AiterFlashAttentionBackend\([^)]*\):\n)",
                   r"\1    @classmethod\n    def supports_head_size(cls, head_size: int) -> bool:\n        return 32 <= head_size <= 256\n\n",
                   s, count=1)
        assert "def supports_head_size" in s, "could not add supports_head_size to rocm_aiter_fa"
        open(fa, "w").write(s)
    rp = os.path.join(root, "platforms/rocm.py")
    s = open(rp).read()
    old = "    if not use_kv_connector:\n        backends.append(AttentionBackendEnum.ROCM_ATTN)\n"
    assert old in s, "ROCM_ATTN priority line not found"
    # Demote ROCM_ATTN to after the AITER backends.
    s = s.replace(old, "")
    s = s.replace("    backends.append(AttentionBackendEnum.TRITON_ATTN)\n    backends.append(AttentionBackendEnum.TURBOQUANT)\n",
                  "    if not use_kv_connector:\n        backends.append(AttentionBackendEnum.ROCM_ATTN)\n    backends.append(AttentionBackendEnum.TRITON_ATTN)\n    backends.append(AttentionBackendEnum.TURBOQUANT)\n")
    open(rp, "w").write(s)
print(f"patched gemma4 attention selection (mode={mode})", file=sys.stderr)
