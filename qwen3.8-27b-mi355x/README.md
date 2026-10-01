# Qwen3.8-27B on AMD MI355X

A vLLM ROCm engine for Qwen3.8-27B (`Qwen/Qwen3.8-27B-FP8`: dense 27B, 48 Gated DeltaNet layers and 16 gated
full-attention layers, FP8 block-scale weights, one MTP layer) on one AMD Instinct MI355X, and the harness that
measured it. It is stock `vllm/vllm-openai-rocm:v0.30.0` plus two tuned config tables and two engine flags.

The production deployment ran `v0.26.0` with a config that left every FP8 GEMM on an untuned default kernel. An
agent-driven tuning loop on one 8-GPU MI355X host (about 3.5 hours, the same method as the Gemma 4 recipe) moved
each GPU from 55K to 148K output tokens per minute at 48 concurrent requests on the production 5:1 workload,
with GSM8K unchanged.

## Results

One MI355X, TP1, same host and harness before and after. Workload: the production 5:1 shape (~2,480 prompt
tokens of unique random text, exactly 500 output tokens, thinking off), closed loop.

**Baseline:** what production runs: `vllm/vllm-openai-rocm:v0.26.0` with the production flags (`BASE_ARGS` in
`bench/engine_args.sh`). Its 55.4K matches production's measured 56.4K output TPM per GPU at 50 concurrent.

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Output tokens/min per GPU at 48 concurrent (primary) | 55.4K | 148.2K | **2.67x** |
| Same, streams desynchronized (closer to production arrivals) | 54.5K | 146.6K | 2.69x |
| Knee: most concurrent streams with TTFT p95 <= 2 s (closed loop) | 4 (14.0K TPM) | 64 (161.6K TPM), the max-num-seqs ceiling | **16x streams, 11.5x TPM** |
| Prefill tokens/s (16 concurrent 8K prompts) | 7.5K | 27.9K | **3.7x** |
| Mean time per output token at 16 / 48 concurrent | 20.4 / 44.6 ms | 10.7 / 18.7 ms | -48% / -58% |
| TTFT p95 at 48 concurrent | 13.0 s | 0.61 to 0.69 s | |
| Cold start to /health 200 (alone, empty cache) / warm restart | 230 s / 69 s | 316 s / 100 s | +86 s / +31 s |
| GSM8K, first 250 test problems, thinking off | 97.2% | 97.2% (98.0% final validation) | unchanged |

Final validation: the shipped stack on a fresh container passed GSM8K (98.0%) and a 5-minute closed-loop soak at
48 concurrent with 1,431 requests and zero errors.

Where each gain came from, one change at a time on one GPU (full table: [results/ablation.md](results/ablation.md)):

| Stage | Prefill tok/s | c48 TPM | c48 TPOT | c16 TPM | Knee |
|---|---|---|---|---|---|
| Baseline A (`v0.26.0`) | 7.5K | 55.4K | 44.6 ms | 36.6K | 4 |
| Stage 1: `v0.30.0`, same flags | 7.3K | 54.5K | 37.6 ms | 37.7K | 4 |
| + tuned FP8 block-scale GEMM rows | 24.8K | 110.2K | 22.2 ms | 62.6K | 16 to 24 |
| + attention prefill tile entry | 28.5K | 111.6K | 21.7 ms | 62.5K | |
| + bf16 GDN recurrent state (`--mamba-ssm-cache-dtype bfloat16`) | 28.6K | 118.2K | 20.0 ms | 64.6K | 16 |
| + fused GDN decode patch (inert once MTP is on; not in the image) | 28.7K | 119.9K | 19.6 ms | 65.5K | 24 |
| + MTP, 2 draft tokens | 27.5K | 140.5K | 19.8 ms | 80.2K | 64 |
| MTP 3 draft tokens instead | 27.9K | 144.6K | 19.2 ms | 88.6K | |
| + exact-M GEMM rows for decode and MTP-verify sizes (shipped) | 27.9K | 148.2K | 18.7 ms | 86.7K | 64 |

`v0.30.0` alone is a wash; it is stage 1 because every later win needs its aiter 0.1.21 kernels and Model Runner V2.

## What is here

| Path | What it is |
|---|---|
| `Dockerfile` | `vllm/vllm-openai-rocm:v0.30.0` plus the two tables below; builds without a GPU |
| `qwen38_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 103 tuned aiter a8w8 block-scale B-preshuffle GEMM rows for the model's five FP8 linear shapes (merged by aiter from `model_configs/`) |
| `unified_attention_gfx950.json` | aiter's gfx950 unified-attention table plus one entry, `attn_2d.D_GEQ_256.Q_GEQ_256.DT_fp8_fp8` (replaces the image's `DEFAULT.json`) |
| `patch_gdn_aiter_decode.py` | Lets Qwen3.5-layout models use aiter's fused GDN decode on ROCm. Only for an MTP-off deployment; not in the image |
| `bench/` | Launcher, stage launcher, gate, load generators, profiling and trace analysis, GEMM tuner scripts ([bench/README.md](bench/README.md)) |
| `results/` | [ablation.md](results/ablation.md), [knee.md](results/knee.md), [profiles.md](results/profiles.md), [exploration.md](results/exploration.md) (everything tried, including what was discarded) |

## Deploy

Engine image: build `Dockerfile` (`docker build -t vllm-rocm-qwen38:v0.30.0-q38mi355 .`), or use the stock
`vllm/vllm-openai-rocm:v0.30.0` with the two files mounted over the image's paths (what the measurements did,
`bench/stack.sh ... gemm2 ua1 ssm16 mtp3`). Not built and pushed to a registry as part of this work.

Environment: `VLLM_ROCM_USE_AITER=1`, `TRITON_CACHE_DIR=/root/.cache/triton` (both set by the image), and
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` as today. Keep `/root/.cache` on a persistent volume.

Flags (production's, plus the last two lines):

```
--model /weights --served-model-name qwen3.8-27b --tensor-parallel-size 1 --data-parallel-size 1
--max-model-len 262144 --max-num-seqs 64 --enable-prefix-caching --enable-prompt-tokens-details
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--kv-cache-dtype fp8 --attention-backend ROCM_AITER_UNIFIED_ATTN
--mamba-ssm-cache-dtype bfloat16
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Rollout checks: the engine log must say `Selected AiterPreshuffledFp8BlockScaledMMKernel for Fp8LinearMethod` (the
GEMM rows are in) and print no `a8w8_blockscale ... not found tuned config`; `/metrics` `vllm:spec_decode_*`
counters give MTP acceptance on real traffic (74.5 / 49.0 / 30.6% per position here).

## What each piece does

**Tuned GEMM rows (2.0x throughput, 3.4x prefill).** Every linear layer is FP8 with 128x128 block scales. On
gfx950 vLLM v0.30 tries `AiterPreshuffledFp8BlockScaledMMKernel` first, but only for (N, K) pairs that have a row
in aiter's B-preshuffle tuned CSV (`vllm/model_executor/kernels/linear/scaled_mm/aiter.py`, `is_blockscale_bpreshuffle_tuned`).
None of this model's shapes had one, so every GEMM fell back to `AiterFp8BlockScaledMMKernel` and CK's default
ABScale instance: 84% of prefill GPU time, ~0.46 PFLOP/s, and 61% of each decode step. aiter's tuner
(`gemm_a8w8_blockscale_tune.py --preshuffle --libtype all`, 11 minutes on 4 GPUs) picked CK B-preshuffle tiles for
small M, aiter ASM kernels around M 128 to 512 and CK-tile 192x256x128 above, reaching 2.0 to 2.35 PFLOP/s at
prefill sizes. A second pass at exact decode and MTP-verify sizes (M 24 to 192) kept 28 rows that beat the
power-of-two row aiter pads to by 5% or more (1.3x to 1.5x at M 144 to 192): +2.5% more at c48.

**Attention prefill entry (+15% prefill).** aiter's gfx950 table is keyed on head dim first, so prefill on the 16
full-attention layers (head_dim 256, fp8 q and KV) landed on `D_GEQ_256`: BLOCK_M 16, which with 6 query heads per
KV head is 2 query tokens per program. v0.26's aiter used BLOCK_M 128 for any prefill, so this was a v0.30
regression. The added entry (BLOCK_M 128, TILE 64, 4 warps, 1 stage) restores it. Decode keeps its tiles; outputs
on short prompts are bit-identical.

**bf16 GDN state (+5% throughput, -9% TPOT).** The checkpoint asks for an fp32 recurrent state: 48 value heads x
128 x 128 x 4 bytes per layer, 144 MiB per request over 48 layers, read and written on every decode step. At 48
requests that is ~13.5 GiB per step next to ~27 GiB of weights. `--mamba-ssm-cache-dtype bfloat16` halves it and
also shrinks the hybrid page, so the attention block size drops from 1600 to 832 tokens. GSM8K unchanged.

**MTP, 3 draft tokens (+21% at c48, +35% at c16, -15% TPOT at c16).** The checkpoint's MTP layer through
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (k > 1 reuses the single layer). Acceptance on
this workload is 74.5 / 49.0 / 30.6% by position (2.54 tokens per step). k=4 was 4% worse. MTP also breaks the
closed-loop lockstep described below, which is why its TTFT drops so much in the closed-loop harness.

## How the tuning was run

Same loop as the Gemma 4 recipe: profile, take the largest slice, smallest experiment that can fail, gate on
GSM8K-250 >= 96.5% (thinking off), keep only if the primary metric improves, one change at a time, journal every
step. Each experiment ran on its own GPU against the previous stage, up to eight at once. A research agent read
the installed vLLM and aiter sources and ranked levers first; the GEMM fallback, the attention tile and the fp32
state were its top three, and all three held.

Mistakes caught and corrected during the run, all in the journal: the first hour ran on a host where another
tenant started an engine on a GPU next to the measurements (results discarded, rerun on a dedicated host); a
windowed throughput count that moved in steps of one closed-loop wave (replaced by Little's law); two benchmark
runs lost to editing a script while bash was executing it.

## Caveats

- **Closed-loop lockstep.** With a fixed 500-token output and no MTP, every stream takes the same time per
  request, so prefills arrive in bursts. That inflates TTFT (A's knee is 4 streams here against ~14.5 per GPU in
  production, which spreads independent arrivals over 11 replicas) and slightly inflates non-MTP throughput. The
  recipe keeps the closed-loop knee definition fixed and reports a desynchronized run next to it.
- **Acceptance is workload-dependent.** The 2.54 tokens per step came from model-written continuations of random
  prompts at temperature 0.7. Check the `/metrics` counters on real traffic; at lower acceptance, k=2 may beat k=3.
- **bf16 GDN state** departs from the checkpoint's `mamba_ssm_dtype: float32`. GSM8K-250 is unchanged and 10 of 20
  temperature-0 smoke answers changed wording; long-context (100K+) quality was not measured.
- **Cold start is 86 s slower** (MTP drafter compile, 51 graph sizes). A new container also rebuilds aiter's
  `mha_varlen_fwd` JIT module (56 to 62 s) because aiter writes JIT modules into site-packages, not `/root/.cache`.
- **One host, one workload shape.** amd-mi355-node-146, ROCm 7.2, one GPU per engine.
- **Version-bound.** Both tables are keyed to aiter `0.1.21.post2` in `v0.30.0`; re-tune when the image moves.

## Next

1. The GDN MTP decode kernel (`fused_sigmoid_gating_delta_rule_update`) is 17% of a decode step at fixed launch
   parameters (BV 32, 4 warps); sweep BV and warps on gfx950, or extend aiter's fused GDN decode to speculative steps.
2. MTP bookkeeping copies are 9% of a decode step.
3. Mixed steps above the 512-token CUDA-graph limit run piecewise (26% idle inside those steps in the trace).
   Capture sizes to 4096 left c48 unchanged and gave +4% at c16 for 2.7 GiB; worth it for a low-concurrency pool.
4. Prebuild `mha_varlen_fwd` into the image to cut ~60 s of cold start.

## Upstream

| Change | Where |
|---|---|
| gfx950 a8w8 block-scale B-preshuffle rows for Qwen3.8-27B (cu_num 256, the 103 rows here) | ROCm/aiter `aiter/configs/model_configs/` |
| `attn_2d.D_GEQ_256.Q_GEQ_256.DT_fp8_fp8` prefill entry | ROCm/aiter `aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json` |
| Don't fall back to the CK default ABScale instance for untuned shapes: use the preshuffled kernel with aiter's heuristic dispatch, or warn loudly | vLLM `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` (`is_blockscale_bpreshuffle_tuned` gate) |
| Qwen3.5-layout fused GDN decode on ROCm (`qkvz_layout="flat"`) | vLLM `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`; already fixed on vLLM main per the nightly source, backport only |
