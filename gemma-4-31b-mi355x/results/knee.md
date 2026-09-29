# Knee: offered load against p95 time to first token

Open loop (`bench/bench.sh`), Poisson arrivals, random-token prompts at the workload's shape (input uniform
1.9K to 17.1K tokens, mean 9.5K; output uniform 76 to 684, mean 380), `DUR=300` seconds per rate, seeds 7
and 8. One MI355X per config, all three measured in the same session. The SLO is p95 TTFT under 4 s; a rate
counts as SLO-safe only if both seeds pass. Input tok/s is the mean of the two seeds.

| Config | req/s | p95 TTFT s (seed 7 / 8) | TPOT p50 ms (seed 7 / 8) | Input tok/s |
|---|---|---|---|---|
| baseline | 1.75 | 2.24 / 2.54 | 27.9 / 27.2 | 16.7K |
| baseline | **2.0** | **2.90 / 2.85** (1 failed request on seed 8) | **43.1 / 39.9** | 18.8K |
| baseline | 2.25 | 5.75 / 5.62 | 65.3 / 66.1 | 20.8K |
| baseline | 2.5 | 25.9 / 27.4 | 69.4 / 69.7 | 21.7K |
| tuned, 16384-token budget | 2.75 | 1.68 / 1.54 | 18.6 / 16.0 | 26.3K |
| tuned, 16384-token budget | 3.0 | 1.81 / 1.86 | 23.9 / 24.7 | 28.6K |
| tuned, 16384-token budget | 3.25 | 3.95 / 5.79 | 41.9 / 45.9 | 30.8K |
| tuned, 16384-token budget | 3.5 | 13.4 / 15.1 | 47.0 / 47.7 | 31.9K |
| tuned (shipped) | 2.75 | 1.59 / 1.42 | 17.3 / 15.1 | 26.3K |
| tuned (shipped) | **3.0** | **1.67 / 1.80** | **21.6 / 22.6** | 28.6K |
| tuned (shipped) | 3.25 | 3.37 / 4.49 | 36.1 / 42.1 | 30.9K |
| tuned (shipped) | 3.5 | 10.7 / 12.8 | 45.6 / 46.7 | **32.2K** |

- **baseline:** stock `vllm/vllm-openai-rocm:v0.30.0`, `BASE_ARGS`.
- **tuned, 16384-token budget:** every tuned piece (both tables, both patches, `G4_FUSE_GELU=1`, graphs to 256
  tokens, native RMSNorm) with the baseline's `--max-num-batched-tokens 16384`. This is ablation stage G.
- **tuned (shipped):** the same with `--max-num-batched-tokens 8192`, i.e. `TUNED_ARGS` on the tuned image.

SLO-safe: baseline 2.0 req/s per GPU, tuned 3.0 (1.5x), with 3.25 on the edge. At the SLO-safe rate TPOT p50
drops from 40 to 43 ms to 22 ms. Saturated input throughput goes from 21.7K to 32.2K tok/s (1.48x). The
8192-token budget mostly buys TPOT: at 3.0 req/s the two tuned configs admit the same load, and the smaller
budget cuts TPOT p50 from 24 to 22 ms.

Random-token prompts give 97 to 98% MTP acceptance per position, far above natural text (see
[exploration.md](exploration.md), "Speculative decoding on natural text"). That flatters absolute TPOT for
every config equally; it does not change which config wins here.

Reproduce: `bench/README.md`, "Knee".
