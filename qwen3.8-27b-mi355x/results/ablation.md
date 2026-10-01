# Ablation: each kept change, cumulative

One MI355X per stage, TP1, amd-mi355-node-146, measured with `bench/measure.sh <tag> <port> full` (stages ran
on different GPUs of the same host; the repeat runs of one config on two GPUs agreed within 1% lockstep).
Every stage passed the gate. Workload and metric definitions: [exploration.md](exploration.md), "Measurement notes".

- **Prefill:** 16 concurrent 8,192-token prompts, 1 output token, second of two runs.
- **c48 / c16:** closed loop, 5:1 shape (~2,480 in, 500 out, thinking off), best of two 120 s runs, output
  tokens/min/GPU by Little's law.
- **Desync:** one run with stream starts spread over one request cycle.
- **Knee:** highest concurrency in {2,4,8,16,24,32,48,64} with TTFT p95 <= 2 s (closed loop), one entry per run.

| Stage | GSM8K 250 | Prefill tok/s | c48 TPM | c48 TTFT p95 | c48 TPOT | c16 TPM | c16 TPOT | c48 TPM desync (TTFT p95) | Knee: conc (TPM) |
|---|---|---|---|---|---|---|---|---|---|
| A: `v0.26.0`, production flags | 97.2% | 7.5K | 55.4K | 13.0 s | 44.6 ms | 36.6K | 20.4 ms | 54.5K (6.0 s) | 4 (14.0K), 4 (13.9K) |
| B: `v0.30.0`, production flags | 97.2% (median of 3) | 7.3K | 54.5K | 14.4 to 14.7 s | 37.6 ms | 37.7K | 16.5 ms | not run | 4 (14.8K), one run |
| + tuned GEMM rows (power of two M) | 97.6% | 24.8K | 110.2K | 2.8 s | 22.2 ms | 62.6K | 12.9 ms | 105.6K (0.47 s) | 16 (62.5K), 24 (80.5K) |
| + attention prefill entry | 97.2% | 28.5K | 111.6K | 2.7 s | 21.7 ms | 62.5K | 13.0 ms | not run | not run |
| + bf16 GDN state | 97.6% | 28.6K | 118.2K | 2.8 to 3.0 s | 20.0 ms | 64.6K | 12.5 ms | 111.7K (0.33 s) | 16 (64.6K), one run |
| + fused GDN decode patch (S) | 98.4% | 28.7K | 119.9K | 2.7 to 2.9 s | 19.6 ms | 65.5K | 12.3 ms | 114.2K (0.48 s) | 24 (85.9K), 24 (85.9K) |
| + MTP k=2 | 96.8% | 27.5K | 140.5K | 0.82 s | 19.8 ms | 80.2K | 11.7 ms | 138.7K (0.40 s) | 64 (159.4K), 64 (159.4K) |
| MTP k=3 instead of 2 | 98.4% | 27.9K | 144.6K | 0.70 s | 19.2 ms | 88.6K | 10.5 ms | 145.6K (0.39 s) | not run |
| + exact-M GEMM rows (F, measured shipped stack) | 97.2%; 98.0% final | 27.9K | 148.2K | 0.61 to 0.69 s | 18.7 ms | 86.7K | 10.7 ms | 146.6K (0.41 s) | 64 (161.6K), 64 (160.6K) |

Notes:

- A's c48 and c16 come from its e2e p50 (its two dedicated runs predate the Little's-law field; in lockstep every
  request takes the same time). Its knee-ladder run, which has the field, read 52.6K at c48.
- B's GSM8K: 96.4, 97.6 and 97.2% at c=32 (98.4% at c=1). Single runs at c=32 vary by about 1 point.
- The F row's c16 (86.7K) is below the k=3 row (88.6K): F's two c16 runs were 85.2K and 86.7K, k=3's were 86.7K
  and 88.6K. Treat c16 differences under 3% as noise; at c48 the two exact-M runs agreed within 0.1%.
- The "MTP k=3 instead of 2" row is S + MTP k=3 with the power-of-two GEMM table; the last row swaps in the
  exact-M table (`bench/stack.sh ... gemm2`).
- With MTP on, the fused GDN decode patch is inert: every decode step carries speculative tokens, and the patched
  aiter path requires `spec_sequence_masks is None`. F's trace shows only the FLA kernels. The shipped image
  therefore does not apply it (`../Dockerfile`); its measured gain (+1.1% c48, -2.7% TPOT on G; -7% TPOT on B)
  applies only to an MTP-off deployment.
- Cold start, each alone from an empty cache: A 230 s, F 316 s; warm restart of the same container: A 69 s,
  F 100 s. F's extra time is the MTP drafter's torch.compile (55 s vs 30 s) and 51 graph sizes instead of 19.

Raw logs: `/mnt/nvme/ops/qwen38-opt/results/<tag>/` on the measurement host (mirrored to the workstation
scratchpad `qwen38-opt/box146/results/`). Tags: A, B, gemm, g-ua1, g-ua1-ssm16, s, s-mtp2, f-mtp3, f.
