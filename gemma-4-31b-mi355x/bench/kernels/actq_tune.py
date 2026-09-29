# actq_tune.py: block-size sweep for aiter's _act_mul_and_dynamic_mxfp4_quant_kernel at the Gemma 4 down_proj input.
import torch, time, triton, itertools
from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
from aiter.ops.triton._triton_kernels.activation import _act_mul_and_dynamic_mxfp4_quant_kernel as K
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for M in [16384, 192, 48]:
    N = 21504; x = torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16)
    ref_q, ref_s = act_mul_and_mxfp4_quant(x, "gelu_tanh", shuffle=True)
    base = timeit(lambda: act_mul_and_mxfp4_quant(x, "gelu_tanh", shuffle=True))
    scaleN_valid = triton.cdiv(N, 32); scaleM = triton.cdiv(M, 256) * 256; scaleN = triton.cdiv(scaleN_valid, 8) * 8
    res = []
    for bm, bn, nw, wpe in itertools.product([32, 64], [64, 128, 256, 512], [2, 4, 8], [0, 1, 2]):
        q = torch.empty((M, N // 2), dtype=torch.uint8, device="cuda"); s = torch.empty((scaleM, scaleN), dtype=torch.uint8, device="cuda")
        def f():
            K[(triton.cdiv(M, bm), triton.cdiv(N, bn))](x, q, s, *x.stride(), *q.stride(), *s.stride(), M=M, N=N,
                MXFP4_QUANT_BLOCK_SIZE=32, SCALING_MODE=0, ACTIVATION="gelu_tanh", scaleN=scaleN_valid, scaleM_pad=scaleM,
                scaleN_pad=scaleN, SHUFFLE=True, NUM_ITER=1, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, NUM_STAGES=1,
                num_warps=nw, waves_per_eu=wpe, num_stages=1)
        try:
            f(); torch.cuda.synchronize()
            ok = torch.equal(q, ref_q) and torch.equal(s[:M], ref_s[:M])
            res.append((timeit(f), bm, bn, nw, wpe, ok))
        except Exception as e:
            pass
    res.sort()
    print(f"M={M} default {base*1000:.1f} us; best: " + "; ".join(f"{t*1000:.1f} us bm{bm} bn{bn} w{nw} wpe{wpe} exact={ok}" for t, bm, bn, nw, wpe, ok in res[:4]), flush=True)
