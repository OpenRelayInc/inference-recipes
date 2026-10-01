# patch_gdn_aiter_decode.py: lets Qwen3.5-layout models (Qwen3.8) use aiter's fused GDN decode on ROCm.
#
# vLLM v0.30.0 routes pure-decode GDN steps on ROCm to aiter's fused reshape + causal-conv1d update + gated
# delta rule kernels only when gqa_interleaved_layout is True (Qwen3-Next). Qwen3.5/3.8 pack in_proj_qkvz as
# [q_all | k_all | v_all | z_all] and ba as [b_all | a_all] ("flat"), so they fall back to the generic path:
# a zero_ of core_attn_out, a z copy, a separate split/rearrange, conv1d update and the FLA packed decode kernel.
# aiter 0.1.21 (the version in the v0.30.0 image) already supports the flat packing through
# qkvz_layout="flat"; this patch drops the layout gate and passes the layout through.
# Locates vLLM without importing it and fails if either anchor is missing.
import pathlib, sys
p = next(pathlib.Path(d) for d in sys.path if (pathlib.Path(d) / "vllm").is_dir())
f = p / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
s = f.read_text()
if "qkvz_layout=" in s and "QWEN38_GDN_PATCH" in s:
    print("patch_gdn_aiter_decode: already applied"); sys.exit(0)
a1 = ("            self.gqa_interleaved_layout\n"
      "            and attn_metadata.spec_sequence_masks is None\n")
a2 = ("                validate_data=True,\n"
      "            )\n"
      "        )\n\n"
      "        # 2. Recurrent attention\n")
assert s.count(a1) == 1, "anchor 1 (layout gate in _forward_core_rocm) not found exactly once"
assert s.count(a2) == 1, "anchor 2 (fused conv1d call in _forward_core_decode_aiter) not found exactly once"
s = s.replace(a1, "            attn_metadata.spec_sequence_masks is None  # QWEN38_GDN_PATCH\n")
s = s.replace(a2, ("                validate_data=True,\n"
                   "                qkvz_layout=\"interleaved\" if self.gqa_interleaved_layout else \"flat\",\n"
                   "            )\n"
                   "        )\n\n"
                   "        # 2. Recurrent attention\n"))
f.write_text(s)
print(f"patch_gdn_aiter_decode: patched {f}")
