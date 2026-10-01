# qtrace.py <trace.json.gz> [topn] [decode|mixed|prefill|all]: GPU kernel time by category for Qwen3.8 (GDN hybrid),
# optionally restricted to vLLM's execute_context_X(T)_generation_Y(U) step windows of one kind, and GPU idle
# percent inside those windows.
import gzip, json, sys, re, collections, bisect
d = json.load(gzip.open(sys.argv[1])); topn = int(sys.argv[2]) if len(sys.argv) > 2 else 15
want = sys.argv[3] if len(sys.argv) > 3 else "all"
ev = d["traceEvents"]
cats = [("attn(full,16L)", r"unified_attention|reduce_segments|paged_attention|fmha|flash_attn"),
        ("gdn_chunk(prefill)", r"chunk_|solve_tril|wy_fast|recompute_w_u|merge_16x16|cumsum|kkt|chunk_fwd|fwd_h|fwd_o"),
        ("gdn_recurrent(decode)", r"fused_recurrent|sigmoid_gating|delta_rule|packed_decode|gated_delta"),
        ("gdn_conv1d", r"causal_conv1d|conv1d"),
        ("gdn_prep/l2norm", r"l2norm|post_conv|fused_gdn|gdn"),
        ("gemm_fp8_blockscale", r"a8w8|blockscale|gemm_xdl|f8|fp8.*gemm|kernel_gemm|ck_tile|_gemm_a8w8"),
        ("gemm_other", r"Cijk|gemm|wvSplitK|mm_|matmul"),
        ("quant", r"quant"), ("norm", r"rms|norm"), ("act(silu*mul)", r"silu|act_and_mul|swiglu"),
        ("kv_cache_write", r"reshape_and_cache|cache"), ("copy/memset", r"copy|memset|memcpy|fill"),]
win = []
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation":
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", e.get("name", ""))
        if not m: continue
        nc, ng = int(m.group(1)), int(m.group(3))
        kind = "decode" if nc == 0 else ("prefill" if ng == 0 else "mixed")
        if want in ("all", kind): win.append((e["ts"], e["ts"] + e["dur"]))
win.sort(); starts = [w[0] for w in win]
def inwin(ts):
    if want == "all" and not win: return True
    i = bisect.bisect_right(starts, ts) - 1
    return i >= 0 and ts < win[i][1]
tot = collections.Counter(); byname = collections.Counter(); cnt = collections.Counter()
for e in ev:
    if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"): continue
    if not inwin(e["ts"]): continue
    n = e["name"]; c = "other"
    for k, p in cats:
        if re.search(p, n, re.I): c = k; break
    tot[c] += e["dur"]; byname[(c, n[:95])] += e["dur"]; cnt[(c, n[:95])] += 1
T = sum(tot.values()); span = sum(b - a for a, b in win)
print(f"[{want}] steps {len(win)}, step window {span/1000:.1f} ms, kernel time {T/1000:.1f} ms"
      + (f", GPU busy {T/span*100:.0f}% (idle {100-T/span*100:.0f}%)" if span else ""))
for k, v in tot.most_common(): print(f"  {k:24} {v/1000:9.2f} ms {v/T*100:5.1f}%")
print("top kernels:")
for (c, n), v in byname.most_common(topn):
    print(f"  {v/1000:8.2f} ms {v/T*100:5.1f}% x{cnt[(c,n)]:<5} [{c}] {n}")
