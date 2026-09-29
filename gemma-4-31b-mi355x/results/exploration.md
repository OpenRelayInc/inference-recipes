# Exploration: every config tried, and the earlier sweeps

These are the measurements taken while searching, before the final method (300 s per rate, two seeds, one
session) was in place. They are kept because they explain the decisions, but read the knee columns with two
caveats:

- Single 150 s runs at 3.0 req/s sit on the cliff for every config and are noisy.
- The GELU fusion first shipped with a Triton JIT-compile storm (about 0.2 s per new batch size, see
  [profiles.md](profiles.md)). Every knee run of a config with the fusion before the fix (cg2, cg3, mtp3,
  nat2, pw1, s128) paid those stalls, mostly in its first rate point. That is why some of them look worse at
  3.0 req/s than at 3.25. Their saturated throughput is slightly pessimistic too.

## Engine configs, correctness gate and prefill probe

Gate: a smoke prompt with a sane answer, and GSM8K (first 250 test problems, c=32) at or above 96.5%.
Prefill: `probe.sh <port> 16 10000 1 64`, run twice; the first run includes JIT warm-up. Configs are
cumulative unless noted.

| Tag | Change | GSM8K 250 | Prefill req/s (run 1, run 2) | Verdict |
|---|---|---|---|---|
| base | baseline (`BASE_ARGS`, stock image) | 247/250 = 98.8% | 2.78, 2.77 (27.8K tok/s) | reference |
| ua1 | + attention prefill entries (`ua_v1.json`) | 247/250 = 98.8% | 3.36, 3.71 (37.1K, 1.33x) | keep |
| gemm1 | ua1 + tuned GEMM rows | 98.8% | 3.88, 3.87 (38.7K, 1.39x) | keep |
| ua2 | d=512 short-query tile added (`ua_v2.json`), compared against ua1 | 245/250 = 98.0% | not reported | neutral at the knee, kept (cuts decode attention kernel time) |
| fuse1 | ua2 + GEMM rows + GELU/MXFP4 fusion | 246/250 = 98.4% | 3.53, 3.94 | small gain, keep |
| cg1 | ua2 + GEMM rows + graphs to 256 tokens | 246/250 = 98.4% | 38.5K in the progression table | keep |
| cg2 | cg1 + fusion | 244/250 = 97.6% | 3.51, 3.94 | keep |
| cg3 | cg2 + split-KV decode attention (`patch_aiter_ua3d.py`, shipped table) | 247/250 = 98.8% | 3.56, 3.98 (39.8K, 1.43x) | keep |
| mtp3 | cg3 with 3 speculative tokens | 246/250 = 98.4% | not reported | optional TPOT tweak, not adopted |
| qkf1 | cg3 + `pass_config {"enable_qk_norm_rope_fusion": true}` | 98.8% | 3.47, 3.90 | no-op for Gemma 4, discard |
| qkf2 | cg3 + `use_inductor_graph_partition` + `fuse_qk_norm_rope_kvcache` | 98.4% | 3.53, 3.96 | no-op for Gemma 4, discard |
| nat1 | cg3 + native `rms_norm` and `fused_add_rms_norm` | 247/250 = 98.8% | 3.64, 4.12 | superseded by nat2 |
| nat2 | cg3 + native `rms_norm` only | 247/250 = 98.8% | 3.68, 4.16 (41.6K, 1.50x); cg3 re-probed alongside: 3.99 | keep |
| pw1 | nat2 + piecewise graphs every 512 tokens to 16384 | not reported | 3.68, 4.12; 14.1 GiB of graph memory | no gain, discard |
| b8k | nat2 + `--max-num-batched-tokens 8192` | 98.8% | 3.72, 4.19 (41.9K) | keep |

qkf2's log shows why the vLLM fusion passes do nothing here: the d=512 layers are unsupported by the AITER
kernel (head dims 64, 128, 256 only), and the pattern matcher "replaced 0 pattern(s)" for the rest, because
Gemma 4's v_norm and K = V layout do not match it.

After the fusion compile fix, nat2 and b8k were relaunched as nat3 and b8k3 for the final runs
([knee.md](knee.md)).

Closed loop, c=48, 9.5K in / 380 out: cg1 3.05 req/s (TPOT 31.1 ms), cg2 3.00 (30.8 ms), cg3 3.13 (30.6 ms).

## Baseline against the attention entries (150 s per rate, seed 7)

| Config | req/s | p95 TTFT s | p50 TTFT s | TPOT p50 ms | TPOT p95 ms | Failed |
|---|---|---|---|---|---|---|
| base | 2.0 | 2.51 | 1.00 | 36.7 | 75.2 | 0 |
| base | 2.5 | 12.01 | 6.86 | 69.5 | 98.6 | 0 |
| base | 3.0 | 43.8 | 24.8 | 67.6 | 94.9 | 0 |
| base | 4.0 | 100.9 | 57.6 | 68.8 | 96.1 | 0 |
| ua1 | 2.0 | 1.43 | 0.50 | 18.7 | 34.9 | 0 |
| ua1 | 2.5 | 1.68 | 0.59 | 28.5 | 49.7 | 0 |
| ua1 | 3.0 | 5.89 | 3.33 | 51.7 | 67.1 | 1 (connection reset, client side) |
| ua1 | 3.5 | 24.6 | 15.7 | 51.8 | 70.8 | 0 |

Saturated input throughput in this sweep is 22K (base) and 28K (ua1) tok/s, well under the pure-prefill probe
(27.8K, 37.1K): decode steps take a large share of the GPU under load.

Other single points: gemm1 p95 TTFT 1.73 s at 2.5 req/s, 1.82 s at 2.75, 3.81 s at 3.0, 10.9 s at 3.25.
ua2 1.75 s at 2.5, 5.91 s at 3.0 (same as ua1). cg1 1.78 s at 2.75, 2.55 s at 3.0.

## Stacked configs (150 s per rate, seed 7)

| Config | req/s | p50 TTFT s | p95 TTFT s | TPOT p50 ms | TPOT p95 ms | Input tok/s |
|---|---|---|---|---|---|---|
| gemm1 | 2.75 | 0.73 | 1.82 | 37.5 | 57.7 | 25.7K |
| gemm1 | 3.0 | 1.68 | 3.81 | 50.2 | 66.0 | 28.1K |
| gemm1 | 3.25 | 6.55 | 10.9 | 51.0 | 69.9 | 28.6K |
| cg1 | 2.75 | 0.75 | 1.78 | 29.0 | 49.8 | 26.0K |
| cg1 | 3.0 | 1.05 | 2.55 | 46.4 | 64.8 | 28.3K |
| cg1 | 3.25 | 6.00 | 9.96 | 50.8 | 67.5 | 29.4K |
| cg1 | 3.5 | 13.1 | 19.9 | 50.0 | 73.1 | 29.5K |
| cg2 | 3.0 | 6.42 | 8.77 | 52.7 | 73.5 | 28.2K |
| cg2 | 3.25 | 4.59 | 8.61 | 49.6 | 70.4 | 29.2K |
| cg2 | 3.5 | 11.4 | 17.5 | 49.6 | 65.4 | 30.1K |
| cg3 | 3.0 | 3.87 | 6.17 | 51.9 | 69.2 | 28.5K |
| cg3 | 3.25 | 3.37 | 6.97 | 49.1 | 66.2 | 30.3K |
| cg3 | 3.5 | 11.8 | 16.7 | 49.3 | 65.1 | 30.3K |
| mtp3 | 3.0 | 6.79 | 9.05 | 50.4 | 81.5 | 28.5K |
| mtp3 | 3.25 | 3.15 | 7.01 | 47.6 | 67.0 | 30.0K |
| mtp3 | 3.5 | 10.4 | 14.8 | 47.6 | 66.6 | 30.4K |

The stable signal is saturated input throughput: ua1 28.2K, gemm1 28.6K, cg1 29.5K, cg2 30.1K, cg3 30.3K,
mtp3 30.4K tok/s. Pure prefill for cg3 is 39.8K, so ~24% of GPU time goes to decode at saturation.

## Native RMSNorm and piecewise graphs (150 s per rate, seed 7)

| Config | req/s | p50 TTFT s | p95 TTFT s | TPOT p50 ms | Input tok/s |
|---|---|---|---|---|---|
| nat2 | 3.0 | 5.01 | 7.96 | 50.3 | 28.4K |
| nat2 | 3.25 | 1.44 | 3.79 | 45.4 | 30.8K |
| nat2 | 3.5 | 7.45 | 11.2 | 46.7 | 31.8K |
| pw1 | 3.5 | 9.25 | 13.1 | 46.9 | 31.3K |

nat2 being worse at 3.0 than at 3.25 led to the JIT-storm finding.

## max-num-seqs 128 (discarded)

At the knee, concurrency is req/s x end-to-end latency = 3.0 x ~20 s = ~60 requests, right at
`--max-num-seqs 64`. Raising the cap (with `BASE3_ARGS`, capture sizes to 384 tokens) tested whether the cap
bounds capacity.

| Config | req/s | p95 TTFT s | TPOT p50 ms | Input tok/s |
|---|---|---|---|---|
| s128 | 3.0 | 3.07 | 95.0 | 28.5K |
| s128 | 3.25 | 4.56 | 91.6 | 30.4K |
| s128 | 3.5 | 13.8 | 95.9 | 30.4K |
| s128 | 3.75 | 18.9 | 95.7 | 31.0K |
| s128 + 8192 budget | 3.0 | 5.20 | 77.7 | 28.5K |
| s128 + 8192 budget | 3.25 | 3.84 | 72.9 | 30.6K |
| s128 + 8192 budget | 3.5 | 12.1 | 85.1 | 31.1K |

Saturated throughput +1 to +3%, TPOT doubles (50 to 95 ms). The 64-sequence cap is not what bounds capacity.

## Speculative decoding on natural text

`bench_text.sh` (prompts from `mkdata.py`, 380 forced output tokens), cg3 with 2 speculative tokens against
mtp3 with 3.

Acceptance on natural text: k=2 mean acceptance length 2.66 to 2.70 (per position 0.87 to 0.90, 0.79 to
0.80); k=3 mean 3.19 to 3.50 (0.82 to 0.89, 0.73 to 0.83, 0.64 to 0.79). On the random-token bench
acceptance is 0.97 to 0.98 per position.

| Config | req/s | p95 TTFT s | TPOT p50 ms |
|---|---|---|---|
| cg3 (k=2), text | 2.5 | 1.19 | 14.6 |
| cg3 (k=2), text | 3.0 | 2.18 | 25.5 |
| cg3 (k=2), text | 3.5 | 9.98 | 48.5 |
| mtp3 (k=3), text | 2.5 | 1.26 | 13.1 |
| mtp3 (k=3), text | 3.0 | 2.22 | 23.6 |
| mtp3 (k=3), text | 3.5 | 9.38 | 48.0 |

k=3 gives 7 to 10% lower TPOT below the knee and the same knee. Not adopted.
