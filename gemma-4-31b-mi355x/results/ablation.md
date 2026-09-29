# Ablation: one change per stage, on one GPU

`bench/ablate.sh`: every stage runs on the same MI355X, so GPU-to-GPU variance cannot confound the per-stage
gains. Each stage adds one change to the stage before it. Per stage: launch, smoke prompt, three prefill
probes (`probe.sh <port> 16 10000 1 64`, runs 2 and 3 reported) and one closed-loop probe at the workload's
shape (`probe.sh <port> 32 9500 380 128`).

| Stage | Change | Prefill req/s (runs 2 to 3) | Prefill tok/s | Closed-loop req/s | Mean TTFT s | Mean TPOT ms |
|---|---|---|---|---|---|---|
| A | baseline | 2.78 to 2.80 | 27.9K | 2.08 | 2.59 | 32.9 |
| B | + attention tile entries (`bench/stages/ua_v1.json`) | 3.73 | 37.3K | 2.32 | 2.08 | 27.9 |
| C | + tuned MXFP4 GEMM rows | 3.87 | 38.7K | 2.42 | 2.19 | 28.7 |
| D | + CUDA graphs to 256 tokens, d=512 short-query tile (`ua_v2.json`) | 3.87 to 3.88 | 38.8K | 2.88 | 2.12 | 22.6 |
| E | + fused GELU*up and MXFP4 quant (`patch_g4_fuse.py`) | 3.93 to 3.94 | 39.4K | 2.94 | 1.95 | 22.6 |
| F | + split-KV decode attention (`patch_aiter_ua3d.py`, shipped table) | 3.95 | 39.5K | 2.98 | 1.96 | 22.4 |
| G | + native RMSNorm (`--kernel-config`) | 4.16 to 4.17 | 41.7K | 3.05 | 1.96 | 21.0 |

Two changes did most of the work, and they fixed different halves of the engine: the attention tile entries
took prefill from 27.9K to 37.3K tok/s (1.34x), and the CUDA-graph capture list took the closed loop from
2.42 to 2.88 req/s (1.19x) by making decode steps GPU-bound. Rounded as in the write-up: GEMM rows +4%
prefill, the GELU fusion +1.5%, native RMSNorm +6%.

Same-GPU A/B of the GEMM table alone on the final stack: 3.99 to 4.00 req/s without it, 4.16 with it (+4%).

The shipped config is stage G plus `--max-num-batched-tokens 8192`; that budget was measured in the knee runs
([knee.md](knee.md)), not here. The 8192-budget variant's prefill probe was 4.19 req/s (41.9K tok/s), so the
smaller budget costs no prefill throughput.

The closed-loop column is the most direct view of a production-shaped load: 2.08 to 3.05 req/s is 1.47x.
