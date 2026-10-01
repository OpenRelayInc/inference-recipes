# ktrace.py <trace.json.gz> [topn]: GPU kernel time by category and top kernels
import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cats = [("attn_global(d512)", r"unified_attention.*head_size_512|unified_attention.*512"),
        ("attn_sliding(d256)", r"unified_attention|fmha|flash|attn"),
        ("gemm_fp4", r"f4gemm|gemm_a4w4|Fp4|fp4"),
        ("gemm_bf16/other", r"Cijk|gemm|wvSplitK|mm_"),
        ("quant", r"quant"), ("norm", r"rmsnorm|norm"),
        ("gelu", r"gelu"), ("kvcache", r"cache"), ]
tot = collections.Counter(); byname = collections.Counter(); cnt = collections.Counter()
for e in ev:
    n = e["name"]; c = "other"
    for k, p in cats:
        if re.search(p, n, re.I): c = k; break
    tot[c] += e["dur"]; byname[n[:110]] += e["dur"]; cnt[n[:110]] += 1
T = sum(tot.values())
print(f"total GPU kernel time {T/1000:.1f} ms")
for k, v in tot.most_common(): print(f"  {k:22} {v/1000:9.1f} ms {v/T*100:5.1f}%")
print("top kernels:")
for n, v in byname.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 25):
    print(f"  {v/1000:8.1f} ms {v/T*100:5.1f}% x{cnt[n]:<5} {n}")
