"""Let aiter unified_attention use the split-KV 3D kernel for head_size >= 512 on short-query (decode / MTP verify)
steps. Stock aiter forces the 2D kernel for D >= 512 whenever max_seqlen_q > 1, which leaves most CUs idle when a
verify step has 3 query tokens per sequence (2D launches num_seqs x num_kv_heads programs, each walking the whole
context). Prefill (max_seqlen_q > 16) keeps the 2D path."""
import os, sys, importlib.util

# locate without importing: aiter probes the GPU on import, and docker build has none
p = os.path.join(os.path.dirname(importlib.util.find_spec("aiter").origin), "ops/triton/attention/unified_attention.py")
s = open(p).read()
old = "    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode:\n"
new = "    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode and params.max_seqlen_q > 16:\n"
if new not in s:
    assert old in s, "aiter use_2d_kernel D>=512 clause not found"
    s = s.replace(old, new, 1); open(p, "w").write(s)
print("patch_aiter_ua3d: applied", file=sys.stderr)
