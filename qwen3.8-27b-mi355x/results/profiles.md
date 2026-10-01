# Profiles: where GPU time went

Engines ran with `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"active_iterations":12}'`.
Load traces: `bench/profload.sh <port> 48 50` (closed-loop c=48 at the production shape, 10 s capture after 50 s).
Prefill traces: `bench/profreq.py <port> 8192 4`. Grouped with `bench/analysis/qtrace.py <trace> 15 decode|mixed|prefill`,
which splits by vLLM's `execute_context_X(T)_generation_Y(U)` step annotations. Traces: `/mnt/nvme/ops/qwen38-opt/traces/`
on the measurement host (B from node-194, G and F from node-146; same image and flags for B on both hosts).

## Decode-only steps at c=48 (kernel time per step, % of step)

| Category | B (v0.30, prod flags) | G (+ tuned GEMM) | F (shipped: G + attention entry + bf16 GDN state + fused GDN decode + MTP k=3) |
|---|---|---|---|
| Step | 48 tokens, 19.4 ms | 48 tokens, 17.6 ms | 191 tokens (48 x 4), 23.3 ms |
| GPU busy | 100% | 100% | 99% |
| FP8 block-scale GEMM | 61.0% (CK default instance) | 55.6% (CK B-preshuffle) | 47.6% (CK B-preshuffle, CK-tile, ASM 96x128) |
| GDN recurrent decode | 13.4% (FLA packed decode, fp32 state) | 14.9% | 17.2% (FLA MTP kernel `fused_sigmoid_gating_delta_rule_update`, bf16 state) |
| Activation quant (per-group fp8) | 7.8% | 7.7% | 6.3% |
| Full attention (16 layers, split-KV 3D + reduce) | 5.8% | 6.1% | 6.9% |
| Copies / fills | 2.6% | 2.9% | 9.1% (MTP bookkeeping: gathers, index copies) |
| RMSNorm, SiLU*mul, other elementwise | ~9% | ~9% | ~9% |
| conv1d update | 1.2% | 1.3% | 1.8% |

Per output token: B 0.40 ms of GPU per token in a decode step, G 0.37 ms, F 0.12 ms per verified token slot and
0.19 ms per emitted token. MTP k=3 acceptance on this workload (engine `/metrics`, `vllm:spec_decode_*` counters):
74.5%, 49.0% and 30.6% at draft positions 1 to 3, 2.54 tokens per request per step. k=2: 74.6% and 49.5%, 2.24.
The workload's output is model-written text continuing random-syllable prompts at temperature 0.7, which is closer
to real traffic than random token ids, but production acceptance should be read from the same counters.

## Mixed steps (prefill chunks + decodes) at c=48

| Category | B | G | F |
|---|---|---|---|
| Avg step | 12.9K prefill + 26 decodes, 1,723 ms | 12.5K prefill + 29 decodes, 409 ms | 2.1K prefill + 46 decodes x 4, 129 ms |
| FP8 GEMM | 87.0% (~0.46 PFLOP/s) | 61.4% (~2.9 PFLOP/s) | 56.3% |
| GDN chunked prefill (FLA Triton) | 3.0% | 11.4% | 10.9% |
| Full attention 2D (prefill) | 1.4% | 5.9% (BLOCK_M 16 tile) | 2.9% (BLOCK_M 128 tile) |
| Quant + SiLU*mul + norm | ~6% | ~10.8% | ~10.5% |
| conv1d | 0.8% | 3.0% | 2.6% |
| Idle inside step windows | ~0% | 0.8% (gaps > 50 us) | 26% (qtrace; see below) |

F's mixed steps are small (MTP desynchronizes the streams, so prefills trickle in instead of arriving in bursts)
and many of them exceed the 512-token CUDA-graph capture limit, so they run piecewise with the GDN core eager
between graphs; qtrace counts 26% of their window as idle. That is the motivation for the `f-cg4k` experiment
(capture sizes to 4096).

## Pure prefill (four 8,192-token prompts)

| Category | B (3 steps, node-194) | F (3 steps) |
|---|---|---|
| Kernel time | 3,395 ms | 930 ms |
| FP8 GEMM | 84.2% | 62.7% |
| Activation quant (incl. fused act+quant) | 5.2% | 4.2% |
| Full attention | 4.3% (BLOCK_M 16, TILE 32) | 5.0% (BLOCK_M 128, TILE 64) |
| GDN chunked prefill | 3.4% | 12.6% |
| conv1d, l2norm, post-conv prep | 1.2% | 4.5% |
| Norms, SiLU*mul, copies | ~1.5% | ~9% |

## Top kernels, before and after (c=48 load, all step kinds)

B (top 5): CK `kernel_gemm_xdl_cshuffle_v3` ABScale (block-scale default instance) 84 to 87% of mixed and 61% of
decode time; `fused_recurrent_gated_delta_rule_packed_decode_kernel` 13% of decode; `kernel_unified_attention_3d`
5%; `dynamic_per_group_scaled_quant_kernel` 2.4%; `_act_mul_and_dynamic_fp8_group_quant_kernel` 2%.

F decode (top 5): CK `kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle` 20.6%; CK-tile
`QuantGemmMultiDKernel` 18.6%; `fused_sigmoid_gating_delta_rule_update_kernel` 17.2%; aiter ASM
`fp8gemm_bf16_blockscale_BpreShuffle_96x128` 8.5%; `kernel_unified_attention_3d` 6.2%.
