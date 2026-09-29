# Day one and the rollout

## The workload

A long-prompt production workload, measured from the serving engines before the move: ~9.5K tokens in on
average (p50 9.4K, p95 17K, p99 22K), ~380 out (p50 370), so about 96% of tokens are prefill. Prefix-cache hit
rate ~5%. Target: p95 time to first token under 4 s at full load. Before the move it ran on NVIDIA B200 (TP1,
vLLM v0.28, `RedHatAI/gemma-4-31B-it-NVFP4` with one MTP draft token from `google/gemma-4-31B-it-assistant`)
at p95 TTFT at or under 0.75 s and ~13 ms between tokens.

## Day one: finding a working MI355X config

NVFP4 executes only on NVIDIA Blackwell, so the MI355X needed different weights.

| Setup | Result |
|---|---|
| RedHatAI's FP8 block-quantized checkpoint | 5.2K prefill tok/s per GPU: aiter's a8w8 blockscale GEMMs have no tuned configs for these shapes |
| MXFP4, vLLM's default attention for Gemma 4 (`TRITON_ATTN`, forced because of the mixed 256/512 head dims) | 10K prefill profile: attention 67% (29.6 ms per global layer, 2.6 ms per sliding layer), MXFP4 GEMM 22% |
| Isolated kernels at the 10K shape | torch SDPA d=512 causal 9.8 ms (against 29.6 in vLLM); AITER CK FA d=256, window 1024: 0.83 ms (against 2.6). CK FA refuses d > 256 |
| MXFP4, CK FA on sliding layers and AITER unified on global layers (`bench/patch_gemma4_attn.py`, split mode) | crashed under batching (HTTP 500s) |
| SGLang ROCm image | garbage output with both aiter and triton attention (day one); in the tuning run it could not load the MXFP4 checkpoint |
| **MXFP4, `ROCM_AITER_UNIFIED_ATTN` on every layer, FP8 KV cache, MTP k=2** | **1.8x over `TRITON_ATTN`; the first working config and the baseline for everything after** |

Also measured on day one: MTP with 2 draft tokens beat 1, which beat none (TPOT and knee), and
`--max-num-batched-tokens` 8K, 16K and 32K made no difference on this config.

The first working config on the first MI355X host: 2.85 req/s pure prefill (`probe.sh` c=16, 10K in) =
28.5K prefill tok/s (~1.8 PFLOP/s effective); open loop p95 TTFT 1.87 / 2.40 / 11.3 s and TPOT p50
17 / 36 / 70 ms at 1.5 / 2.0 / 2.5 req/s; GSM8K on 500 problems 98.0% (the B200 deployment: 97.0%).
The tuning run used a different MI355X host that measured the same config ~2.5% slower (27.8K tok/s); every
comparison in this directory uses that second host.

## Rollout of the tuned config

- Before merge, the tuned image was A/B'd against the first config on one GPU under the production
  container's resource limits: 1.54x prefill, 1.51x closed loop, TPOT 33.5 to 21.5 ms.
- It went out as a blue/green revision on 18 MI355X GPUs: smoke test on all 18, then the full 1,319-problem
  GSM8K (97.3% tuned, 97.1% before), then a traffic ramp and cutover.
- At 80% of the traffic on the tuned config: p95 TTFT at or under 0.75 s (the same as B200), 14 ms between
  tokens (B200: about 13), no errors, no queueing.

This is not a B200 against MI355X benchmark. The tuned harness never ran on B200.
