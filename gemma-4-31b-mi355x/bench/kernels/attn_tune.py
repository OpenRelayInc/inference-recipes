# attn_tune.py: sweep aiter unified_attention 2d configs at vLLM's real Gemma 4 prefill shapes.
# Paged fp8 KV cache in vLLM's LBHNC layout, fp8 query (vLLM quantizes q for fp8 KV), scale 1.0.
# usage: python3 attn_tune.py <d: 512|256> [scenario: 10k|16k|mix] [quick]
import sys, time, itertools, json, os, torch, triton
import aiter.ops.triton.attention.unified_attention as UA
torch.manual_seed(0)
D = int(sys.argv[1]); scen = sys.argv[2] if len(sys.argv) > 2 else "10k"
quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
HQ = 32; HKV = 4 if D == 512 else 16; BS = 128 if D == 512 else 64
window = (-1, -1) if D == 512 else (1023, 0)
fp8 = torch.float8_e4m3fn
# sequences as (q_len, k_len): a 16K-token prefill batch like the scheduler builds
seqs = {"10k": [(10000, 10000), (6384, 6384)], "16k": [(16000, 16000)],
        "mix": [], "dec": []}[scen]
if scen == "dec":  # decode-only MTP step: 48 requests x 3 query tokens at ~9.5K context
    seqs = [(3, 9500)] * 48
if scen == "mix":  # 40 decode requests with 3 spec tokens at ~9.5K context
    seqs = [(9000, 9000), (7000, 17000)] + [(3, 9500)] * 40
qlens = [s[0] for s in seqs]; klens = [s[1] for s in seqs]
T = sum(qlens); nblk_per = [triton.cdiv(k, BS) for k in klens]; NB = sum(nblk_per) + 8
kv = (torch.randn(NB, HKV, BS, 2 * D, device="cuda") * 0.5).to(fp8)
kc, vc = kv.transpose(1, 2).split(D, dim=-1)
bt = torch.zeros(len(seqs), max(nblk_per), dtype=torch.int32, device="cuda")
perm = torch.randperm(NB - 8, device="cuda").int(); o = 0
for i, n in enumerate(nblk_per): bt[i, :n] = perm[o:o + n]; o += n
q = (torch.randn(T, HQ, D, device="cuda")).to(fp8)
cu = torch.tensor([0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device="cuda")
sk = torch.tensor(klens, dtype=torch.int32, device="cuda")
one = torch.ones((), device="cuda")
def flops():
    f = 0
    for ql, kl in seqs:
        for i in range(ql):
            ctx = kl - ql + i + 1
            f += min(ctx, 1024) if D == 256 else ctx
    return 4 * f * HQ * D
F = flops() if T < 20000 else None
orig = UA.get_unified_attention_config
override = {}
def patched(op, params, backend="triton", arch=None):
    c = orig(op, params, backend, arch)
    if op == "attn_2d" and override: c = dict(override)
    return c
UA.get_unified_attention_config = patched
def run():
    out = torch.empty(T, HQ, D, device="cuda", dtype=torch.bfloat16)
    UA.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=max(qlens), seqused_k=sk,
        max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=window, block_table=bt,
        softcap=0, q_descale=one, k_descale=one, v_descale=one)
    return out
def bench(n=5):
    run(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): run()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
ref = run(); base = bench()
print(f"D={D} scen={scen} T={T} default: {base:.2f} ms" + (f"  {F/base/1e9:.0f} TFLOP/s" if F else ""), flush=True)
grid = dict(BLOCK_M=[32, 64, 128, 256], TILE_SIZE=[32, 64, 128], num_warps=[4, 8], num_stages=[1, 2], waves_per_eu=[1, 2])
if scen == "dec": grid = dict(BLOCK_M=[16, 32, 64], TILE_SIZE=[16, 32, 64, 128], num_warps=[2, 4], num_stages=[1, 2], waves_per_eu=[1, 2])
if quick: grid = dict(BLOCK_M=[64, 128], TILE_SIZE=[32, 64], num_warps=[4, 8], num_stages=[1], waves_per_eu=[1])
res = []
for vals in itertools.product(*grid.values()):
    cfg = dict(zip(grid, vals))
    if cfg["BLOCK_M"] < (8 if D == 512 else 2) or cfg["BLOCK_M"] > 128: continue
    override.clear(); override.update(cfg)
    try:
        o = run(); torch.cuda.synchronize()
        err = (o.float() - ref.float()).abs().max().item()
        ms = bench()
        res.append((ms, cfg, err))
        print(f"{ms:8.2f} ms {base/ms:5.2f}x err={err:.4f} {cfg}", flush=True)
    except Exception as e:
        print(f"   fail {cfg}: {str(e)[:120]}", flush=True)
    torch.cuda.empty_cache()
res.sort(key=lambda r: r[0])
print("BEST:"); [print(f"{ms:8.2f} ms {base/ms:5.2f}x err={e:.4f} {c}") for ms, c, e in res[:8]]
