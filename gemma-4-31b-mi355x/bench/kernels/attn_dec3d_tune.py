# attn_dec3d_tune.py: sweep split-KV 3D kernel configs for D=512 MTP verify steps (n x 3 queries, ~9.5K ctx).
import sys, time, itertools, torch, triton
import aiter.ops.triton.attention.unified_attention as UA
torch.manual_seed(0)
fp8 = torch.float8_e4m3fn
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
D, HQ, HKV, BS = 512, 32, 4, 128
orig_cfg = UA.get_unified_attention_config; ov = {}
def patched(op, params, backend="triton", arch=None):
    c = orig_cfg(op, params, backend, arch)
    if op in ov: c.update(ov[op])
    if op == "kv_split" and "NUM_SEGMENTS_FIXED" in ov.get("x", {}): c["NUM_SEGMENTS"] = ov["x"]["NUM_SEGMENTS_FIXED"]
    return c
UA.get_unified_attention_config = patched
orig2d = UA.use_2d_kernel
def setup(nseq):
    qlens = [3] * nseq; klens = [9500 + 37 * i for i in range(nseq)]
    T = sum(qlens); nblk = [triton.cdiv(k, BS) for k in klens]; NB = sum(nblk) + 8
    kv = (torch.randn(NB, HKV, BS, 2 * D, device="cuda") * 0.5).to(fp8)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    bt = torch.zeros(nseq, max(nblk), dtype=torch.int32, device="cuda")
    perm = torch.randperm(NB - 8, device="cuda").int(); o = 0
    for i, n in enumerate(nblk): bt[i, :n] = perm[o:o + n]; o += n
    q = torch.randn(T, HQ, D, device="cuda").to(fp8)
    cu = torch.tensor([0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device="cuda")
    sk = torch.tensor(klens, dtype=torch.int32, device="cuda"); one = torch.ones((), device="cuda")
    def run():
        out = torch.empty(T, HQ, D, device="cuda", dtype=torch.bfloat16)
        return UA.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=3, seqused_k=sk,
            max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1), block_table=bt,
            softcap=0, q_descale=one, k_descale=one, v_descale=one)
    return run, sum(klens) * HKV * D
grid = [dict(BLOCK_M=bm, num_warps=nw, num_stages=ns, waves_per_eu=wpe, TILE=t, SEG=seg)
        for bm in [32] for nw in [2, 4] for ns in [1, 2] for wpe in [1, 2] for t in [64, 128] for seg in [4, 8, 16, 32]]
for nseq in [8, 16, 32, 48, 64]:
    run, kvb = setup(nseq)
    UA.use_2d_kernel = orig2d; ov.clear(); ref = run().clone(); t2 = timeit(run)
    UA.use_2d_kernel = lambda p: False
    best = []
    for g in grid:
        ov.clear()
        ov["attn_3d"] = dict(BLOCK_M=g["BLOCK_M"], num_warps=g["num_warps"], num_stages=g["num_stages"], waves_per_eu=g["waves_per_eu"])
        ov["kv_split"] = dict(TILE_SIZE=g["TILE"]); ov["x"] = dict(NUM_SEGMENTS_FIXED=g["SEG"])
        try:
            o3 = run(); err = (o3.float() - ref.float()).abs().max().item(); t3 = timeit(run)
            best.append((t3, err, g))
        except Exception as e:
            pass
    best.sort(key=lambda x: x[0])
    print(f"nseq={nseq:3} 2d {t2*1000:6.1f} us ({kvb/t2/1e6:5.2f} TB/s)  best3d {best[0][0]*1000:6.1f} us ({kvb/best[0][0]/1e6:5.2f} TB/s) err {best[0][1]:.4f} {best[0][2]}", flush=True)
    for b in best[1:4]: print(f"          {b[0]*1000:6.1f} us err {b[1]:.4f} {b[2]}")
UA.use_2d_kernel = orig2d
