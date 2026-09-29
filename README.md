# inference-recipes

Serving configurations we run in production at [OpenRelay](https://openrelay.inc), with the measurements
behind them and the harness to reproduce those measurements. Each recipe pins a model, a GPU, an engine image
and its arguments, and states its baseline.

| Recipe | Model | GPU | Engine | Result |
|---|---|---|---|---|
| [gemma-4-31b-mi355x](gemma-4-31b-mi355x/) | Gemma 4 31B, AMD Quark MXFP4 + MTP drafter | AMD Instinct MI355X, TP1 | vLLM `v0.30.0` ROCm, two tuned aiter tables, two patches | 1.5x requests per GPU under a 4 s p95 TTFT against our first working MI355X config; GSM8K unchanged |

## How a recipe is laid out

```
<model>-<gpu>/
  README.md    results with the baseline stated, how to build, run and reproduce, what each piece does, caveats
  Dockerfile   the engine image: a pinned upstream image plus the recipe's files
  ...          config tables and source patches the Dockerfile copies in
  bench/       launcher, correctness gate, load generators, profiling, microbenchmarks, and the exact
               commands behind every table
  results/     the measured tables, as markdown (CSV where it is simple)
```

Every throughput number in a recipe comes from a config that passed a correctness gate, on the same host and
harness as its baseline.

## License

Apache-2.0. See [LICENSE](LICENSE).
