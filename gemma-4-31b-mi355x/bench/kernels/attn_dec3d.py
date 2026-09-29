# attn_dec3d.py: MTP verify-step attention (n seqs x 3 query tokens at ~9.5K context), 2D kernel (what aiter
# picks for D>=512 when max_seqlen_q>1) vs the split-KV 3D kernel + reduce. Uses the ua_v2 table if mounted.
import sys, time, itertools, torch, triton
import aiter.ops.triton.attention.unified_attention as UA
torch.manual_seed(0)
fp8 = torch.float8_e4m3fn
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
orig2d = UA.use_2d_kernel
for D in [512, 256]:
    HQ = 32; HKV = 4 if D == 512 else 16; BS = 128 if D == 512 else 64
    window = (-1, -1) if D == 512 else (1023, 0)
    for nseq in [4, 8, 16, 32, 48, 64]:
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
                max_seqlen_k=max(klens), softmax_scale=D ** -0.5, causal=True, window_size=window, block_table=bt,
                softcap=0, q_descale=one, k_descale=one, v_descale=one)
        UA.use_2d_kernel = orig2d; ref = run().clone(); t2 = timeit(run)
        UA.use_2d_kernel = lambda p: False
        try:
            o3 = run(); err = (o3.float() - ref.float()).abs().max().item(); t3 = timeit(run)
        except Exception as e:
            err = float("nan"); t3 = float("nan"); print("3d fail", str(e)[:200])
        kvbytes = sum(min(k, 1024 if D == 256 else k) for k in klens) * HKV * D * (1 if D == 512 else 2)
        print(f"D={D} nseq={nseq:3}  2d {t2*1000:7.1f} us ({kvbytes/t2/1e9:5.0f} GB/s)  3d {t3*1000:7.1f} us ({kvbytes/t3/1e9:5.0f} GB/s)  maxerr {err:.4f}", flush=True)
UA.use_2d_kernel = orig2d
