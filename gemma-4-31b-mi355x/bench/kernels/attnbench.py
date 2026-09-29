# attnbench.py: day-one isolated attention kernels at the Gemma 4 10K prefill shape (torch SDPA for
# head_dim 512, CK flash attention for the head_dim 256 sliding window), used to size the attention gap.
import torch, time, traceback
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"
L, HQ, HKV = 10000, 32, 4

def timeit(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

for D, window in [(512, None), (256, 1024)]:
    q = torch.randn(L, HQ, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(L, HKV if D == 512 else 16, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    hkv = k.shape[1]
    flops = 4 * L * (L if window is None else window) * HQ * D / (2 if window is None else 1)
    print(f"== D={D} window={window} hkv={hkv} (~{flops/1e12:.2f} TFLOP)")
    cu = torch.tensor([0, L], device=dev, dtype=torch.int32)
    ref = None
    try:
        import aiter
        ws = (-1, -1) if window is None else (window - 1, 0)
        f = lambda: aiter.flash_attn_varlen_func(q, k, v, cu, cu, L, L, causal=True, window_size=ws)
        ms = timeit(f); out = f(); out = out[0] if isinstance(out, tuple) else out
        ref = out
        print(f"aiter.flash_attn_varlen_func: {ms:.2f} ms  {flops/ms/1e9:.0f} TFLOP/s")
    except Exception as e:
        print("aiter FA failed:", repr(e)[:300])
    try:
        qs, ks, vs = q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        if window is None:
            f = lambda: F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
        else:
            i = torch.arange(L, device=dev)
            mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)
            f = lambda: F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, enable_gqa=True)
        ms = timeit(f)
        o = f()[0].transpose(0, 1)
        err = (o.float() - ref.float()).abs().max().item() if ref is not None else float("nan")
        print(f"torch SDPA: {ms:.2f} ms  {flops/ms/1e9:.0f} TFLOP/s  maxdiff-vs-aiter {err:.4f}")
    except Exception as e:
        print("SDPA failed:", repr(e)[:300])
