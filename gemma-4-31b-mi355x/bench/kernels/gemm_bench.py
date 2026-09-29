# gemm_bench.py: MXFP4 GEMM backends at Gemma 4 shapes: aiter asm gemm_a4w4 (as vLLM calls it),
# aiter Triton gemm_afp4wfp4, and torch._scaled_mm (hipBLASLt) if it accepts MXFP4.
import torch, time, sys
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
torch.manual_seed(0)
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
shapes = [(16384, 5376), (5376, 8192), (43008, 5376), (5376, 21504), (20480, 5376), (5376, 16384)]
Ms = [int(m) for m in sys.argv[1:]] or [8192, 16384]
for M in Ms:
    for N, K in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        wq, ws = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
        fl = 2 * M * N * K
        res = {}
        # asm path exactly as vLLM: weight shuffled (16,16), scale swizzled
        sm, sn = ws.shape
        ws_sw = ws.view(torch.uint8).view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
        wq_sh = shuffle_weight(wq, layout=(16, 16))
        def asm():
            xq, xs = aiter.per_1x32_f4_quant_hip(x, shuffle=True)
            return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
        xq, xs = aiter.per_1x32_f4_quant_hip(x, shuffle=True)
        def asm_only():
            return aiter.gemm_a4w4(xq, wq_sh.view(xq.dtype), xs, ws_sw.view(xs.dtype), dtype=torch.bfloat16, bpreshuffle=True)
        res["asm+quant"] = timeit(asm); res["asm"] = timeit(asm_only)
        try:
            from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            wsT = ws.view(torch.uint8).T.contiguous()
            xq2, xs2 = dynamic_mxfp4_quant(x)
            y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            res["triton"] = timeit(lambda: gemm_afp4wfp4(xq2, wq.view(torch.uint8), xs2, wsT.T, torch.bfloat16, y))
        except Exception as e:
            res["triton"] = float("nan"); print("triton fail", str(e)[:200])
        print(f"M={M:6} N={N:6} K={K:6} " + "  ".join(f"{k} {v:7.3f} ms {fl/v/1e9:5.0f} TF/s" for k, v in res.items()), flush=True)
