# Results

Qwen3.8-27B-FP8, TP1 on one MI355X, amd-mi355-node-146, `vllm/vllm-openai-rocm:v0.26.0` (baseline) and
`v0.30.0` (every later stage). Workload: the production 5:1 shape (~2,480 prompt tokens of unique random text,
exactly 500 output tokens, thinking off), closed loop on one GPU.

| File | What it holds |
|---|---|
| [ablation.md](ablation.md) | One kept change per stage, cumulative, all five metrics |
| [knee.md](knee.md) | Closed-loop ladders 2 to 64 streams for the baseline and five stages, and why the harness knee differs from production's |
| [profiles.md](profiles.md) | Kernel-time tables for decode, mixed and prefill steps, before (B), after GEMM tuning (G) and shipped (F) |
| [exploration.md](exploration.md) | Every config tried, discarded ones included, and the measurement notes (Little's law, lockstep, gate noise) |

Raw logs (per-run `tpm_bench.py` output, prefill probes, GSM8K, smoke answers) and the torch traces are on the
measurement host under `/mnt/nvme/ops/qwen38-opt/{results,traces}` and mirrored with the run journal to the
workstation scratchpad (`qwen38-opt/box146/`).

## Headline

| Metric | Baseline (v0.26.0, production flags) | Shipped | Change |
|---|---|---|---|
| Output TPM per GPU at 48 concurrent | 55.4K | 148.2K | 2.67x |
| Knee (TTFT p95 <= 2 s, closed loop) | 4 streams, 14.0K TPM | 64 streams, 161.6K TPM | |
| Prefill tok/s, 16 x 8K prompts | 7.5K | 27.9K | 3.7x |
| TPOT mean at 16 / 48 concurrent | 20.4 / 44.6 ms | 10.7 / 18.7 ms | |
| Cold start / warm restart | 230 / 69 s | 316 / 100 s | |
| GSM8K 250 | 97.2% | 97.2%, 98.0% | |

## Distance from the hardware

- Decode at c48 (F): 23.3 ms per step for 48 requests x 4 token slots, 99% GPU busy. FP8 GEMMs take 11.1 ms of it;
  ~25 GB of FP8 linear weights (the engine's 29 GiB also holds the vision tower and embeddings) in 11.1 ms is
  ~2.3 TB/s against ~8 TB/s HBM peak, so decode GEMMs have up to ~3x headroom on bandwidth alone. The GDN MTP
  decode kernel (4.0 ms) and MTP bookkeeping copies (2.1 ms) are the next-largest slices.
- Prefill (F): the GEMMs run at ~2.3 PFLOP/s (tuner measurement at M >= 2048, cktile kernels), about 46% of the
  nominal ~5 PFLOP/s dense FP8 peak (clocks under load were not logged), and are 63% of prefill time; GDN chunked
  prefill is 13%, attention 5%.
