# Profiles: where GPU time went

Engines ran with the torch profiler enabled:
`--profiler-config {"profiler":"torch","torch_profiler_dir":"/root/.cache/prof","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"active_iterations":3}`.
Prefill traces come from `bench/profreq.py <port> 10000 3` (three 10K prompts, which the scheduler runs as two
steps of 16384 and 13616 tokens) and `profreq.py <port> 16000 2`, grouped with `bench/analysis/ktrace.py`.
Load traces come from `bench/profload.sh` (20 s window), split per scheduler step with
`analysis/steps.py` and `analysis/stepkern.py` using vLLM's `execute_context_X(T)_generation_Y(U)` GPU
annotations.

## Prefill, baseline

| Category | 10K (1121 ms GPU) | 16K (1344 ms GPU) |
|---|---|---|
| Global attention, d=512 (10 layers, `kernel_unified_attention_2d ... HEAD_SIZE_512`) | 28.3% (14.4 ms per layer-step) | 39.1% |
| MXFP4 GEMM (`f4gemm_bf16_per1x32Fp4_BpreShuffle_{64x1024,256x256,224x256}`) | 40.1% | 32.9% |
| Sliding attention, d=256 (50 layers, 1.4 ms per layer-step) | 14.7% | 13.1% |
| Activation quant (per-1x32 fp4, q fp8) | 5.7% | 5.1% |
| GELU*mul (Triton, fused) | 4.1% | 3.6% |
| Other (rmsnorm+quant, adds, KV-cache write, embedding, lm_head) | 7% | 6.3% |

Effective rates in the 10K step: global attention ~375 TFLOP/s, sliding ~390 TFLOP/s (both ~15% of the BF16
peak), MXFP4 GEMM ~4.6 PFLOP/s (~46% of the 10 PFLOP/s MXFP4 peak).

Root cause for attention: aiter's gfx950 table
(`aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json`) has `D` as its first
axis, so every prefill with head_size 512 and fp8 q + fp8 KV lands on `D_GEQ_512.DT_fp8_fp8`: BLOCK_M 16,
TILE 32, num_warps 4, num_stages 2. With 8 queries per KV head, BLOCK_M 16 is 2 query tokens per program, so
each K/V tile is re-read for every 2 tokens. The prefill-sized entry (`Q_GEQ_256`: BLOCK_M 128, TILE 64) is
never reached for d >= 256. Sliding d=256 lands on `D_GEQ_256` (BLOCK_M 16, TILE 32, 2 warps). Attention was
43% (10K) to 52% (16K) of GPU time, running a decode-shaped tile.

## Prefill, after each change (same three 10K prompts)

| Config | GPU ms | MXFP4 GEMM | Global attn | Sliding attn | Act quant | GELU / fused act+quant | Norms | Other |
|---|---|---|---|---|---|---|---|---|
| baseline | 1121 | 40.1% | 28.3% | 14.7% | 5.7% | 4.1% | | 7% |
| + attention entries (ua1) | 829 | 55% | 13.5% | 8.4% | 7.7% | 5.6% | | 9.6% |
| + GEMM rows, graphs, fusion, split-KV decode (cg3) | 782 | 58.3% | 14.4% | 8.7% | | 6.8% | | ~12% |
| + native RMSNorm (nat2) | 742 | 61.0% | 15.5% | 9.4% | | 7.2% | 6.1% | ~1% |

- ua1: global attention 317 to 112 ms (14.4 to 5.1 ms per layer-step, 2.8x), sliding 165 to 70 ms (2.4x).
- cg3: the remaining small ops (`analysis/kparent.py`): `aten::copy_` of q/k after the split (q [T,32,256],
  k [T,16,256], global q [T,32,512]) ~24 ms per 30K tokens; q/k/v per-head RMSNorm on [T*heads, 256|512]
  ~29 ms; q fp8 static quant ~10 ms; hip activation quant for qkv/o ~15 ms; RoPE (`triton_poi_fused_3`) 13 ms;
  KV-cache write 14 ms. Together ~100 ms, 13% of prefill GPU time.
- nat2: the q/k `aten::copy_` kernels and the per-head aiter rmsnorm launches are gone. Inductor emits
  `triton_red_fused_2` (split + q/k norm + RoPE, 35 ms), `triton_red_fused_add_mul_rms_norm_1` (21 ms) and
  `triton_red_fused_add_rms_norm_0` (16 ms).

The same data by part, in milliseconds, is what the write-up's prefill chart shows. It applies each
profile's category shares to its measured total, so it is derived, not measured per part:

| Config | Attention | MXFP4 matmuls | Everything else | Total |
|---|---|---|---|---|
| baseline | 482 | 450 | 189 | 1121 |
| tuned (nat2) | 185 | 453 | 104 | 742 |

## Decode steps under load

Open loop at 2.5 req/s, ua2 + GEMM rows (graphs captured only to 64 tokens):

- 38 mixed steps (avg 9.6K prefill tokens + 32 decode requests): avg 262 ms GPU window.
- 146 decode-only steps (avg 32.9 requests x 3 tokens = 99 tokens): **57.7 ms window, 21.3 ms of kernels
  (37% busy)**. Decode-only steps are 46% of the annotated GPU window.
- Cause: `cudagraph_capture_sizes` stopped at 64 tokens, and with MTP k=2 each decoding request is 3 tokens,
  so any decode batch above 21 requests ran eagerly (piecewise, no graph) and was launch-bound.
- Kernel time inside a decode step: global attention 6.85 ms (10 layers), sliding attention 3.31 ms
  (50 layers), MXFP4 GEMMs ~5.6 ms, rmsnorm/quant/elementwise ~4 ms.

Same load after capturing graphs to 256 tokens (cg1): decode-only steps **100% GPU-busy, 20.1 ms per step**
(avg 14 requests; median 18.4 ms below 8 requests, 21.3 ms at 24 to 31 requests). Median TPOT at 2.5 req/s
30.5 ms (was 41.6 ms). Kernel split per decode step: global attention 6.4 ms (32%), MXFP4 GEMMs ~4.7 ms,
sliding attention 2.4 ms, norm/quant/elementwise ~4.5 ms. Capture adds 7 s of startup and 3.1 GiB.

## Near saturation (cg3 at 3.25 req/s, 20 s window)

- 54 mixed steps (avg 10.6K prefill tokens + 62 decode requests) = 89% of GPU time, avg 310 ms, only 92%
  busy: eager mode above the 256-token graph sizes costs ~26 ms per step of launch gaps. Kernel split per
  mixed step: MXFP4 GEMM ~135 ms (48%), global attention 48.8 ms, sliding 26.2 ms, fused act+quant 19.2 ms,
  rmsnorm/quant/copies ~45 ms.
- 82 decode-only steps (avg 64 requests) = 11%, 25.8 ms each: global split-KV attention 6.2 ms, sliding decode
  6.2 ms, GEMMs ~6.6 ms, rest ~6.7 ms.

## The compile storm in the GELU fusion

`analysis/gaps.py` on the cg3 saturation trace: 8.3% of mixed-step time is idle, and 99% of that is 7 gaps of
~190 ms, each between the gate_up GEMM and `_act_mul_and_dynamic_mxfp4_quant_kernel`. These are Triton
compiles. aiter declares `scaleM_pad = cdiv(M, 256) * 256` as `tl.constexpr`, so every new token count rounded
to 256 compiles a new variant (64+ variants up to 16K tokens), ~0.2 s each, in the middle of serving. It only
feeds a mask comparison. `patch_g4_fuse.py` makes it a runtime argument. Verified: after one compile per
`EVEN_M_N` variant (M = 16384 and 9000), M = 5000, 300 and 700 run without compiling (0.1 to 0.4 ms first call).

Triton's cache defaults to `/root/.triton`, outside a mounted cache directory, so every restart recompiles.
The image sets `TRITON_CACHE_DIR=/root/.cache/triton`; mount a persistent volume at `/root/.cache`.
