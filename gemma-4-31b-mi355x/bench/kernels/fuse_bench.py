# fuse_bench.py: does aiter's Triton act_mul_and_mxfp4_quant(shuffle=True) produce inputs the asm gemm_a4w4
# accepts, matching vLLM's unfused path (Inductor gelu_tanh*mul, then per_1x32_f4_quant_hip), and is it faster?
import torch, time
import torch.nn.functional as F
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
torch.manual_seed(0)
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
N, K = 5376, 21504
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
wq, ws = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
sm, sn = ws.shape
ws_sw = ws.view(torch.uint8).view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
wq_sh = shuffle_weight(wq, layout=(16, 16))
act = torch.compile(lambda gu: F.gelu(gu[:, :K], approximate="tanh") * gu[:, K:])
for M in [16384, 9000, 144, 3]:
    gu = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    def ref():
        h = act(gu)
        xq, xs = aiter.per_1x32_f4_quant_hip(h, shuffle=True)
        return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
    def fused():
        xq, xs = act_mul_and_mxfp4_quant(gu, "gelu_tanh", shuffle=True)
        xq = xq.view(aiter.dtypes.fp4x2); xs = xs.view(aiter.dtypes.fp8_e8m0)
        return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
    yr, yf = ref(), fused()
    exact = (act(gu).float() @ w.float().T)
    rel = lambda y: ((y.float() - exact).norm() / exact.norm()).item()
    print(f"M={M:6} ref {timeit(ref):.3f} ms  fused {timeit(fused):.3f} ms  relerr ref {rel(yr):.4f} fused {rel(yf):.4f}  ref-vs-fused {((yr.float()-yf.float()).norm()/yr.float().norm()).item():.4f}", flush=True)
