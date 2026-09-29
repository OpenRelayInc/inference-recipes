# Results

Gemma 4 31B (`amd/gemma-4-31B-it-MXFP4` plus the `google/gemma-4-31B-it-assistant` MTP drafter), TP1 on one
MI355X, the same GPU host and harness before and after. The workload shape: ~9.5K tokens in (uniform 1.9K to
17.1K), ~380 out.

**Baseline:** our first working MI355X config, stock `vllm/vllm-openai-rocm:v0.30.0` with `BASE_ARGS` from
`../bench/engine_args.sh`. Not B200. **Tuned:** `TUNED_ARGS` with this recipe's tables and patches, either
baked into the image built from `../Dockerfile` or mounted and applied at start on the stock image (the form
used in the run; the files are byte-identical).

| File | What it holds |
|---|---|
| [knee.md](knee.md), [knee.csv](knee.csv) | The final comparison: open-loop sweeps, 300 s per rate, two seeds |
| [ablation.md](ablation.md), [ablation.csv](ablation.csv) | One change per stage, cumulative, on one GPU |
| [exploration.md](exploration.md) | Every engine config tried, and the earlier single-seed sweeps |
| [profiles.md](profiles.md) | Where GPU time went: prefill kernel splits, decode steps under load, saturation |
| [kernels.md](kernels.md) | Kernel microbenchmarks: attention tiles, split-KV decode, MXFP4 GEMM tuning and backends, fusions |
| [migration-and-rollout.md](migration-and-rollout.md) | Day-one config search and the production rollout |

## Measured gain versus baseline

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Pure prefill, `probe.sh` c=16, 10K in | 27.8K tok/s (2.78 req/s) | 41.9K tok/s (4.19 req/s) | **1.51x** |
| SLO-safe req/s per GPU (p95 TTFT < 4 s on both seeds, 300 s x 2 seeds) | 2.0 | 3.0 (3.25 borderline) | **1.5x** |
| TPOT p50 at the SLO-safe rate | 40 to 43 ms | 22 ms | **-47%** |
| Saturated input tok/s, open loop | 21.7K | 32.2K | 1.48x |
| Closed loop, c=32, 9.5K in / 380 out, same GPU (ablation stage G) | 2.08 req/s, TPOT 32.9 ms | 3.05 req/s, TPOT 21.0 ms | 1.47x |
| GSM8K, first 250 problems, c=32 | 98.8% | 98.8% | unchanged |

At 3.25 req/s the tuned config passes on one seed and misses on the other (p95 TTFT 3.37 s and 4.49 s), so
the SLO-safe gain is 1.5x, up to 1.6x. The closed-loop row comes from the ablation, which stops before the
8192-token prefill budget.

## Progression during the run

The best config at each point of the run, in the order the changes were found. Single 150 s knee runs are
noisy at the cliff, and every config with the GELU fusion before its compile fix paid mid-serving JIT
stalls, so only the last two rows use the final method (300 s x 2 seeds).

| Config | Prefill tok/s (probe c=16, 10K) | SLO-safe req/s | TPOT p50 at that rate | GSM8K 250 | vs baseline |
|---|---|---|---|---|---|
| Baseline | 27.8K | 2.0 (p95 2.85 to 2.90 s, 2 seeds x 300 s) | 40 to 43 ms | 98.8% | 1.00x |
| + attention tile entries | 37.1K | 2.5 (150 s run) | 28.5 ms | 98.8% | 1.33x prefill |
| + tuned MXFP4 GEMM rows | 38.7K | 3.0 borderline (3.81 s) | 50 ms | 98.8% | 1.39x |
| + CUDA graphs to 256 tokens | 38.5K | 3.0 (2.55 s) | 46 ms | 98.4% | 1.39x |
| + GELU/MXFP4 fusion, split-KV decode attention | 39.8K | about 3.0 (noisy, JIT storm) | 25 ms (natural-text bench) | 98.8% | 1.43x |
| + native RMSNorm, fusion compile fix | 41.5K | 3.0 (1.81 / 1.86 s, 2 seeds x 300 s) | 24 to 25 ms | 98.8% | 1.49x / 1.5x |
| **+ 8192-token prefill budget (shipped)** | **41.9K** | **3.0 (1.67 / 1.80 s); 3.25 borderline (3.37 / 4.49 s)** | **22 ms** | **98.8%** | **1.51x prefill, 1.5x knee** |

## Distance from the hardware ceiling

- Gemma 4 31B text: 60 layers, hidden 5376, FFN 21504. 50 sliding-window layers (head_dim 256, 16 KV heads,
  window 1024), 10 global layers (head_dim 512, 4 KV heads, K = V). 32 query heads.
- Prefill FLOPs at a 10K prompt: ~59 GFLOP/token of MXFP4 linears plus ~5 GFLOP/token of attention. 41.9K
  tok/s is ~2.5 PFLOP/s on the linears, ~25% of the nominal 10 PFLOP/s MXFP4 dense peak.
- Under load the GPU runs at ~1850 to 1950 MHz (1160 to 1210 W of 1400 W), so the clock-adjusted MXFP4 peak
  is ~7.7 PFLOP/s. The aiter assembly GEMMs run at 4.1 to 4.7 PFLOP/s (55 to 61% of that), and GEMMs are 61%
  of prefill GPU time. Attention runs at ~1 PFLOP/s (global, d=512) and ~0.8 PFLOP/s (sliding), 25% of time.
  Elementwise, norm and quant are ~14%.
- Upper bounds from the tuned profile: attention and elementwise at zero cost give ~68K tok/s (1.6x more);
  GEMMs at 100% of the clock-adjusted peak on top of that give ~125K tok/s. A realistic next tier (attention
  2x faster, elementwise halved) is ~50K to 55K tok/s, another 1.2 to 1.3x. Ten times the day-one 28.5K
  (285K tok/s) would need ~17 PFLOP/s on the linears alone, above the MXFP4 peak.
- The knee tracks capacity: 3.0 to 3.25 req/s per GPU is ~0.93x the saturated 3.4 req/s. About 24% of GPU
  time at saturation is decode (MTP verify steps).
