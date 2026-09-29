# fuse_bench2.py: aiter Triton fused_rms_mxfp4_quant(shuffle=True) feeding the asm gemm_a4w4 vs vLLM's
# unfused path (aiter rmsnorm then per_1x32_f4_quant_hip). Shapes: hidden 5376 -> qkv 16384 and gate_up 43008.
import torch, time
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_rms_mxfp4_quant
torch.manual_seed(0)
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
K = 5376
nw = (torch.rand(K, device="cuda", dtype=torch.bfloat16) + 0.5)
for N in [16384, 43008]:
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, ws = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
    sm, sn = ws.shape
    ws_sw = ws.view(torch.uint8).view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
    wq_sh = shuffle_weight(wq, layout=(16, 16))
    for M in [16384, 9000, 144, 3]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 3
        def ref():
            h = aiter.rms_norm(x, nw, 1e-6)
            xq, xs = aiter.per_1x32_f4_quant_hip(h, shuffle=True)
            return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
        def fused():
            (xq, xs), _, _, _ = fused_rms_mxfp4_quant(x, nw, 1e-6, shuffle=True)
            xq = xq.view(aiter.dtypes.fp4x2); xs = xs.view(aiter.dtypes.fp8_e8m0)
            return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
        yr, yf = ref(), fused()
        hx = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * nw.float())
        exact = hx @ w.float().T
        rel = lambda y: ((y.float() - exact).norm() / exact.norm()).item()
        print(f"N={N} M={M:6} ref {timeit(ref):.3f} ms  fused {timeit(fused):.3f} ms  relerr ref {rel(yr):.4f} fused {rel(yf):.4f}", flush=True)
