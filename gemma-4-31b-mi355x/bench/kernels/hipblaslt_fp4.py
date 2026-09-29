# hipblaslt_fp4.py: can torch._scaled_mm (hipBLASLt) run MXFP4 x MXFP4 with e8m0 1x32 block scales on gfx950, and how fast?
import torch, time
def timeit(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
print(torch.__version__, hasattr(torch, "float4_e2m1fn_x2"), torch.cuda.get_device_name())
for M, N, K in [(16384, 43008, 5376), (16384, 5376, 21504), (16384, 16384, 5376)]:
    a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    try:
        f = lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)
        ms = timeit(f)
        print(f"M={M} N={N} K={K}: {ms:.3f} ms  {2*M*N*K/ms/1e9:.0f} TFLOP/s")
    except Exception as e:
        print("scaled_mm fail:", str(e)[:300])
