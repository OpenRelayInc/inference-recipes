# Kernel microbenchmarks

Scripts in `bench/kernels/`, run inside the engine image on one GPU (commands in `bench/README.md`).

## Attention tiles, prefill (`attn_tune.py`)

The harness calls `aiter.ops.triton.attention.unified_attention` exactly as vLLM does: paged fp8 KV cache in
the LBHNC layout (page 128 for d=512, 64 for d=256), fp8 query, a 16384-token step made of sequences
(10000, 10000) and (6384, 6384). It overrides only the `attn_2d` config. Error is max-abs against the default
config's output.

| Head dim | Default | Best | Speedup | Notes |
|---|---|---|---|---|
| 512 (global), quick grid | 10.59 ms (436 TFLOP/s) | BLOCK_M 128, TILE 64, w4, s1, wpe1: 4.09 ms (err 0.015) | 2.59x | BLOCK_M 128, TILE 32: 5.63 ms (err 0.002); BLOCK_M 64, TILE 64: 6.65 ms |
| 512 (global), full grid | 10.64 ms | BLOCK_M 128, TILE 64, w4, s2, wpe1: 3.89 ms; s1: 3.92 ms | 2.74x | BLOCK_M 256 is 3x to 9x slower (spills); num_warps 8 always worse |
| 256 (sliding, window 1024), full grid | 1.37 ms (377 TFLOP/s) | BLOCK_M 128, TILE 64, w4, s1, wpe2: 0.59 ms (~870 TFLOP/s) | 2.30x | BLOCK_M 128, TILE 32, w4, s2, wpe2: 0.63 ms (err 0.002) |

The d=256 sweep aborted at BLOCK_M 256, TILE 32 on an LLVM "Bad machine code" error (a Triton compiler bug),
so the harness caps BLOCK_M at 128. The error difference between TILE 32 and 64 (0.002 against 0.015) comes
from the fp8 P path with a different online-softmax tile; it passed the GSM8K gate.

16K single-sequence and mixed prefill+decode scenarios confirm the same entries (d=512 2.68x and 2.64x,
d=256 1.80x).

Decode-shaped step (`attn_tune.py 512 dec`, 48 sequences x 3 query tokens at 9.5K context, the MTP verify
shape): default 0.62 ms, BLOCK_M 32, TILE 64, w4, s2, wpe1 0.42 ms (1.47x). The same for d=256 gave only 1.11x
and was not added.

## Split-KV decode attention for d=512 (`attn_dec3d.py`, `attn_dec3d_tune.py`)

aiter's `use_2d_kernel` forces the 2D kernel for head_size >= 512 whenever `max_seqlen_q > 1`. MTP verify
steps have 3 query tokens per sequence, so a 14-request decode step launches 14 x 4 = 56 programs on 256 CUs,
each walking ~9.5K tokens of context.

| Sequences x 3 queries, ~9.5K context | Stock 2D | Tuned split-KV 3D |
|---|---|---|
| 8 | 361 us | 85 us |
| 16 | 379 us | 160 us |
| 32 | 409 us | 288 us |
| 48 | 452 us | 395 us |
| 64 | 875 us | 650 us |

Max error 0.001. Best 3D config: `attn_3d` BLOCK_M 32, num_warps 2, num_stages 1, waves_per_eu 1, TILE 64,
4 to 16 segments. Through the engine's own selection logic with the shipped table and patch, n = 4, 8, 16,
32, 48, 64 take 81, 80, 172, 334, 395, 641 us. Even tuned, the 3D kernel reads KV at only ~2 to 2.6 TB/s of
the 8 TB/s HBM bandwidth.

## MXFP4 GEMM tuning (aiter tuner, `g4_a4w4_untuned.csv`)

aiter's `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py --compare` over the six (N, K) shapes the
engine logged as "not found tuned config", times M in 16 to 16384 (66 shapes, ~15 min on one GPU).

- Gains only on the N=5376 shapes (o_proj sliding K=8192, o_proj global K=16384, down K=21504): 1.2x to 1.6x
  at M 16 to 512 and M >= 8192 (at M=16384: down 953 to 772 us, o_proj 415 to 332 us), plus
  (M, N, K) = (1024, 20480, 5376) at 1.76x.
- The large-N shapes (qkv 16384 and 20480, gate_up 43008) are already on the best assembly kernel
  (~4.7 PFLOP/s at M=16384).
- The 44 rows that improved by 3% or more are `gemma4_a4w4_blockscale_tuned_gemm.csv`. aiter's
  `get_padded_m` maps any M > 4096 to the 8192 row, so they cover every prefill chunk size.
- In the engine: +4.6% prefill over the attention entries alone (probe 3.71 to 3.87 req/s), +4% on the final
  stack (3.99 to 4.00 req/s without, 4.16 with).

## MXFP4 GEMM backends (`gemm_bench.py`, `hipblaslt_fp4.py`)

At M=16384 on the six Gemma 4 shapes:

| Backend | Rate |
|---|---|
| aiter assembly `gemm_a4w4` (what vLLM v0.30.0 uses on gfx950) | 4.1 to 4.7 PFLOP/s |
| aiter Triton `gemm_afp4wfp4` | 1.7 to 1.9 PFLOP/s (2.5x slower on every shape) |
| hipBLASLt via `torch._scaled_mm` (float4_e2m1fn_x2, e8m0 1x32 scales) | 1.7 to 2.4 PFLOP/s |

The separate hip activation quant costs 0.04 to 0.23 ms per GEMM at M=16384 (up to 20% of the down_proj GEMM).

Clocks: under a prefill load the GPU drew 1163 W of its 1400 W cap at 1854 MHz. At that clock the MXFP4 peak
is ~10 x 1854 / 2400 = ~7.7 PFLOP/s, so the assembly GEMMs at 4.7 PFLOP/s run at ~61% of the clock-adjusted
peak.

## Fusions around the MXFP4 activations

- GELU(tanh)*up + MXFP4 quant feeding the assembly down_proj GEMM (aiter Triton
  `act_mul_and_mxfp4_quant(gate_up, "gelu_tanh", shuffle=True)`): relative error against fp32 0.183 fused,
  0.191 unfused, on random data. Activation + quant + down GEMM per layer: M=16384 1.516 to 1.377 ms; M=144
  0.079 to 0.055 ms. Shipped as `patch_g4_fuse.py`. (The microbenchmark for this one was not preserved.)
- RMSNorm + MXFP4 quant (`fuse_bench2.py`, aiter `fused_rms_mxfp4_quant`): 0 to 3% at prefill and slower at
  decode. Not pursued.
- Block sizes of the fused GELU+quant kernel (`actq_tune.py`): the default (BLOCK_M 32, BLOCK_N 256, 4 warps)
  is already best.
