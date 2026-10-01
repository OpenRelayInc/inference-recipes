# Exploration: every config tried

All runs: one MI355X per engine, TP1, Qwen3.8-27B-FP8, production flags (`BASE_ARGS` in `../bench/engine_args.sh`)
plus the change under test. Workload: `tpm_bench.py`, ~2,480 prompt tokens of unique random text, exactly 500
output tokens (`ignore_eos`), thinking off, temperature 0.7, closed loop, 20 s warmup + 120 s window. TPM is
output tokens per minute per GPU from Little's law (`lit_tpm`, see "Measurement notes"). Prefill is 16
concurrent 8,192-token prompts with 1 output token (`vllm bench serve`, second of two runs). Gate: GSM8K first
250 test problems at c=32, thinking off, >= 96.5%.

Unless marked, every number below was taken on amd-mi355-node-146 (whole box ours). The first hour ran on
amd-mi355-node-194 beside another tenant's engine; those numbers are in `../bench/` history only and are not used.

## Baselines

| Tag | Config | GSM8K 250 | Prefill tok/s | c48 TPM | c48 TTFT p95 | c48 TPOT | c16 TPM | c16 TPOT | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | `v0.26.0` + production flags (what production runs) | 97.2%, 97.2% | 7.5K | 55.4K (from e2e) | 13.0 s | 44.6 ms | 36.6K | 20.4 ms | baseline to beat |
| B | `v0.30.0` + production flags | 96.4%, 97.6%, 97.2% (c=32); 98.4% (c=1) | 7.3K | 54.5K | 14.4 to 14.7 s | 37.6 ms | 37.7K | 16.5 ms | base for every experiment |
| C | `v0.30.0`, no `--attention-backend`, no fp8 KV (engine defaults; node-194) | 97.2% | 6.3K | 48,000 (out_tpm) | 15.7 s | 41.7 ms | not measured | | discard: ROCM_ATTN default and bf16 KV are slower, half the KV capacity (3.6M vs 7.1M tokens) |

On its own, the v0.30.0 image is a wash against v0.26.0 (lower TPOT, slightly lower prefill, same throughput). It
is still stage 1 because every win below needs v0.30.0's aiter 0.1.21 (the B-preshuffle ASM/CK-tile block-scale
GEMMs, the flat-layout fused GDN decode, Model Runner V2 with MTP).

## Single changes on B

| Tag | Change | GSM8K | Prefill | c48 TPM | c48 TPOT | c16 TPM | Verdict |
|---|---|---|---|---|---|---|---|
| gemm (G) | tuned aiter a8w8 block-scale B-preshuffle GEMM rows | 97.6% | 24.8K (3.4x) | 110.2K (2.0x) | 22.2 ms | 62.6K | KEEP |
| ua1 | `D_GEQ_256.Q_GEQ_256.DT_fp8_fp8` attention entry | 96.8% | 7.6K | 60,000 (out_tpm) | 37.0 ms | 40,000 (out_tpm) | keep, re-measured on G |
| ssm16 | `--mamba-ssm-cache-dtype bfloat16` | 97.2% | 7.4K | 60,000 (out_tpm) | 33.6 ms | 39,500 (out_tpm) | keep, re-measured on G |
| gdn | `patch_gdn_aiter_decode.py` | 98.4% | 7.4K | 55.6K | 35.0 ms | 37.9K | keep, re-measured on G |
| mtp1 | MTP, 1 draft token | 98.0% | 7.2K | 55.5K | 45.1 ms | 44.6K | re-measured on G |
| nopc | no `--enable-prefix-caching` | 97.6% | 7.3K | 54.4K | 36.7 ms | 37.5K | DISCARD: no effect; production keeps it |

## Single changes on G (gemm)

| Tag | Change | GSM8K | Prefill | c48 TPM | c48 TTFT p95 | c48 TPOT | c16 TPM | Smoke vs G | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| G | reference | 97.6% | 24.8K | 110.2K | 2.83 s | 22.2 ms | 62.6K | | |
| g-ua1 | + attention entry | 97.2% | 28.5K (+15%) | 111.6K | 2.70 s | 21.7 ms | 62.5K | 0/20 | KEEP |
| g-ssm16 | + bf16 GDN state | 97.6% | 24.7K | 116.2K (+5.4%) | 3.15 to 3.42 s | 20.1 ms | 64.5K | 10/20 | KEEP |
| g-gdn | + fused GDN decode patch | 97.6% | 24.9K | 111.4K (+1.1%) | 2.88 s | 21.6 ms | 63.1K | 11/20 | KEEP (small, consistent in both runs) |
| g-mtp1 | + MTP k=1 | 97.6% | 24.2K | 128.9K (+17%) | 0.89 to 0.98 s | 21.5 ms | 80.4K | 11/20 | KEEP, tune k on the stack |

## The stack S = G + ua1 + ssm16 + gdn, and single changes on it

| Tag | Change | GSM8K | Prefill | c48 TPM lockstep / desync | c48 TTFT p95 lockstep / desync | c48 TPOT | c16 TPM lockstep / desync | Verdict |
|---|---|---|---|---|---|---|---|---|
| S | reference | 98.4% | 28.7K | 119.9K / 114.2K | 2.74 to 2.94 / 0.49 s | 19.6 ms | 65.5K / 62.6K | |
| s-mtp1 | MTP k=1 | 96.8% | 27.8K | 135.8K / 127.7K | 0.99 to 1.09 / 0.36 s | 20.2 ms | 78.3K / 77.6K | superseded |
| s-mtp2 | MTP k=2 | 96.8% | 27.5K | 140.5K / 138.7K | 0.82 / 0.40 s | 19.8 ms | 80.2K / 79.5K | KEEP, then k=3 |
| s-mbt8k | `--max-num-batched-tokens 8192` | 97.2% | 27.8K | 119.5K / 113.1K | 1.36 to 1.62 / 0.33 s | 21.7 ms | 65.4K / 63.2K | DISCARD: no throughput change, TPOT +10% |

## Single changes on S + MTP k=2

| Tag | Change | GSM8K | Prefill | c48 TPM lockstep / desync | c48 TPOT | c16 TPM lockstep / desync | Smoke | Verdict |
|---|---|---|---|---|---|---|---|---|
| s-mtp2 | reference | 96.8% | 27.5K | 140.5K / 138.7K | 19.8 ms | 80.2K / 79.5K | | |
| f-gemm2 | 28 exact-M GEMM rows (M 40 to 192) | 97.6% | 27.6K | 144.5K / 141.2K | 19.3 ms | 82.1K / 81.5K | 0/20 | KEEP (+2 to 3%) |
| f-mbt8k | `--max-num-batched-tokens 8192` | 97.2% | 27.2K | 141.1K / 139.7K | 19.7 ms | 81.2K / 80.1K | 0/20 | DISCARD: within noise |
| f-mtp3 | MTP k=3 | 98.4% | 27.9K | 144.6K / 145.6K | 19.2 ms | 88.6K / 88.1K | 9/20 | KEEP (+3 to 5% at c48, +10% at c16) |

## Single changes on F = S + gemm2 + MTP k=3 (the shipped stack)

| Tag | Change | GSM8K | c48 TPM | c48 TPOT | c16 TPM | Verdict |
|---|---|---|---|---|---|---|
| F | reference | 97.2% | 148.2K | 18.7 ms | 86.7K | |
| f-mtp4 | MTP k=4 | 97.6% | 142.5K (-4%) | 19.5 ms | 85.6K | DISCARD |
| f-cg4k | CUDA-graph capture sizes extended from 512 to 4096 tokens | 97.2% | see README | | | see README |

## Levers checked in source and not run

- **CUDA-graph capture sizes to 64+ (Gemma's biggest decode win).** Already covered: v0.30.0's default list for
  `--max-num-seqs 64` goes to 128 tokens without MTP and to 512 with MTP k=3, with FULL graphs for uniform decode
  steps (GDN backend reports `UNIFORM_BATCH`, aiter unified attention `ALWAYS`). Decode steps profile at 99 to
  100% GPU busy on B, G and F.
- **Native RMSNorm (`--kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}'`).** The flag exists in v0.26 and
  v0.30, but changes nothing here: Qwen3.5's `GemmaRMSNorm` passes an fp32 `(1+w)` weight, aiter's `rms_norm`
  rejects a weight dtype that differs from x, so the native (Inductor) path already runs. The gated RMSNorm in
  the GDN block runs `forward_native` under Inductor by default.
- **Per-batch-size recompiles (Gemma's constexpr storm).** None found: the FLA kernels run varlen with B=1 and
  constant strides; the only batch-dependent constexpr is aiter attention's `NUM_SEGMENTS` (a few power-of-two
  values). No mid-serving compile gaps in the traces (gaps.py: 0.8% of mixed-step time on G).
- **GDN Triton autotune configs on gfx950.** The chunked-prefill kernels autotune over HIP-specific lists; the
  decode kernels have fixed launches (packed decode: BV 32, 1 warp; MTP decode: BV 32, 4 warps). The MTP decode
  kernel is 17% of F's decode step and is the next target (see README, "Next").
- **`--gpu-memory-utilization` and KV block size.** KV capacity is not the constraint (7.1M fp8 tokens at 0.92;
  64 x 3K-token sequences need 0.2M). The attention block size is forced to the mamba page size (1600 tokens with
  fp32 GDN state, 832 with bf16), not settable independently.

## Measurement notes

- **Wave quantization.** `out_tpm` counts output tokens of requests that finished inside the 120 s window. With a
  fixed 500-token output, a closed loop finishes requests in waves of up to 48, so `out_tpm` moves in steps of
  about 2,000 (B read 60,000 while its sustained rate was 54.5K). `lit_tpm` (concurrency x mean completion tokens
  / mean e2e) has no such step and is the number used everywhere. A's runs predate the field; its value comes
  from its e2e p50, which equals the mean in lockstep (all requests take the same time).
- **Lockstep.** Without MTP every stream takes the same time per request, so streams that start together stay
  together and their prefills arrive as bursts. That inflates TTFT and also inflates throughput (prefill runs as a
  few large, efficient mixed steps). MTP's variable acceptance breaks the lockstep by itself. The `desync` runs
  (`STAGGER=12` at c48, `STAGGER=4` at c16) start stream i at i x S / n and are the better predictor of production,
  where arrivals are independent. Desync runs are single runs; repeat measurements of G differed by 6% (99.3K
  and 105.6K on two engines), so treat desync deltas under 5% as noise.
- **Gate noise.** GSM8K-250 at c=32 is not batch-invariant at temperature 0: baseline B scored 96.4, 97.6 and
  97.2% on three runs. `smoke20.py` (one request at a time) is deterministic and is the strict numerics check.
- **Cold start in parallel launches.** Ready times of 245 to 660 s in this table's runs came from up to eight
  engines compiling at once on a 236-core host. Cold start is measured alone in `../README.md`.
