# Knee: closed-loop ladders

`bench/tpm.sh <port> <log> 2,4,8,16,24,32,48,64` per run: each step is a closed loop of N streams at the 5:1
production shape (~2,480 prompt tokens of unique random text, exactly 500 output tokens, thinking off), 20 s warmup,
120 s window. Cells: output TPM per GPU (Little's law) / TTFT p95. The knee is the highest N with TTFT p95 <= 2 s.

| N | A: v0.26 prod, run 1 | A run 2 | B: v0.30 prod | G: + tuned GEMM, run 1 | G run 2 | S, run 1 | S run 2 | S + MTP k=2, run 1 | run 2 | F (shipped), run 1 | F run 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 7.5K / 0.72 s | 7.7K / 0.72 s | 8.2K / 0.72 s | 10.5K / 0.24 s | 10.5K / 0.26 s | 10.8K / 0.28 s | 10.7K / 0.25 s | 17.4K / 0.25 s | 17.6K / 0.19 s | 18.5K / 0.19 s | 18.6K / 0.20 s |
| 4 | 14.0K / 1.38 s | 13.9K / 1.38 s | 14.8K / 1.39 s | 20.0K / 0.40 s | 20.0K / 0.40 s | 20.8K / 0.39 s | 20.8K / 0.39 s | 31.4K / 0.27 s | 31.6K / 0.20 s | 33.1K / 0.27 s | 33.4K / 0.25 s |
| 8 | 24.0K / 2.52 s | 23.3K / 2.54 s | 24.6K / 2.76 s | 35.9K / 0.75 s | 35.8K / 0.76 s | 37.3K / 0.73 s | 37.2K / 0.74 s | 53.8K / 0.29 s | 53.4K / 0.30 s | 56.5K / 0.28 s | 57.1K / 0.27 s |
| 16 | 36.0K / 3.83 s | 36.2K / 3.85 s | 37.6K / 5.23 s | 62.5K / 1.34 s | 62.5K / 1.34 s | 65.5K / 1.40 s | 65.5K / 1.40 s | 80.5K / 0.34 s | 80.9K / 0.36 s | 87.5K / 0.34 s | 87.0K / 0.33 s |
| 24 | 43.3K / 4.06 s | 43.3K / 4.05 s | 44.8K / 7.64 s | 80.4K / 2.12 s | 80.5K / 1.98 s | 85.9K / 2.00 s | 85.9K / 2.00 s | 98.1K / 0.43 s | 98.9K / 0.41 s | 109.4K / 0.38 s | 109.1K / 0.38 s |
| 32 | 51.0K / 7.97 s | 47.9K / 7.96 s | 49.9K / 10.1 s | 96.7K / 2.62 s | 96.9K / 2.62 s | 105.0K / 2.59 s | 105.1K / 2.60 s | 116.5K / 0.50 s | 118.0K / 0.54 s | 124.3K / 0.48 s | 123.3K / 0.51 s |
| 48 | 58.0K / 12.9 s | 52.6K / 13.0 s | 55.0K / 14.3 s | 110.0K / 2.78 s | 110.4K / 2.82 s | 119.6K / 2.74 s | 119.6K / 3.05 s | 140.7K / 0.93 s | 140.5K / 0.92 s | 149.1K / 0.72 s | 146.0K / 0.61 s |
| 64 | 58.0K / 18.0 s | 55.4K / 18.0 s | 57.8K / 20.3 s | 133.2K / 3.33 s | 133.2K / 2.83 s | 145.5K / 4.84 s | 147.8K / 4.38 s | 159.4K / 0.85 s | 159.4K / 0.94 s | 161.6K / 0.90 s | 160.6K / 0.83 s |
| **Knee** | **4 (14.0K)** | **4 (13.9K)** | **4 (14.8K)** | 16 (62.5K) | 24 (80.5K) | 24 (85.9K) | 24 (85.9K) | **64 (159.4K)** | 64 | **64 (161.6K)** | **64 (160.6K)** |

A run 1 predates the Little's-law field; its cells are the windowed count (`out_tpm`). B has one clean run on this
host (a second was lost to a script edit mid-run, see the journal). G+ua1+ssm16 run 1: knee 16 (64.6K), 2.01 s at 24.

## Why the harness knee is not production's 14.5 per GPU

Production measured about 14.5 concurrent per GPU at TTFT p95 2.0 s with A's config, spreading independent
client arrivals over 11 replicas behind a router. This harness runs a closed loop on one GPU with a fixed
500-token output, so without MTP every stream takes the same time per request and the streams stay in lockstep:
their prefills arrive together and queue behind each other, which puts A over 2 s at 8 streams. The recipe keeps
the closed-loop definition fixed and reads only relative change from it. The desync runs (`STAGGER`, see
[exploration.md](exploration.md)) show the size of the effect: at c48, A's TTFT p95 falls from 13.0 s (lockstep) to
6.0 s, and G's from 2.8 s to 0.24 to 0.47 s. MTP's variable acceptance desynchronizes the streams on its own, so
its lockstep and desync numbers agree (F: 0.61 to 0.69 s lockstep, 0.41 s desync at c48).

At 64 streams, the maximum `--max-num-seqs`, F still has TTFT p95 under 1 s. The ladder does not go higher
because production runs at most 64 sequences per engine.
