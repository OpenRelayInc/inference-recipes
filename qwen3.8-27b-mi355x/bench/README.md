# Benchmark and correctness harness

The scripts behind every number in `../results/`, adapted from `gemma-4-31b-mi355x/bench`. Run them on the GPU
host from this directory. Engines run in containers named `qopt-*` with pod-equivalent limits (`--memory 300g
--cpus 32 --shm-size 8g`, IPC_LOCK, memlock unlimited, groups 44 and 992), exactly one GPU's render and card nodes
plus `/dev/kfd`, the weights read-only, and the port published on 127.0.0.1 only.

## Setup

`common.sh` holds the defaults; override from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `WORK` | `/mnt/nvme/ops/qwen38-opt` | Results, per-engine caches (`cache/<container>` at `/root/.cache`), traces, data |
| `WEIGHTS` | `/var/lib/openrelay/weights/8369...c978` | `Qwen/Qwen3.8-27B-FP8` (platform weights ref `qwen3.8-27b-fp8`) |
| `CLIENT_IMAGE` | `vllm/vllm-openai-rocm:v0.30.0` | Runs `vllm bench serve` and `tpm_bench.py` (needs aiohttp), CPU only |
| `MODEL` | `qwen3.8-27b` | Served model name |
| `WORDS` | `1195` | Random-syllable words per `tpm_bench.py` prompt; calibrated to 2,468 to 2,495 prompt tokens |

Host specifics this harness handles: the docker daemon runs with `userns-remap` (container root is host uid
100000, so `launch.sh` chowns each cache dir to it), and `--network host` is refused under userns-remap, so the
CPU-only client containers run with `--userns=host --network host`. GPU n maps to `renderD(128+8n)` and
`card(8n)`; check `/sys/class/drm/*/device` on a new host.

## Scripts

| Script | What it does |
|---|---|
| `engine_args.sh` | `PROD_ARGS` (production flags minus attention/KV), `BASE_ARGS` (production flags), `DEFAULT_ARGS` |
| `launch.sh <name> <gpu> <port> <image> [args]` | One TP1 engine; `EXTRA_MOUNTS`, `EXTRA_ENV`, `PRE`, `CACHE`. Refuses a busy port or a GPU another `qopt-*` container holds |
| `stack.sh <name> <gpu> <port> [features] [-- args]` | `v0.30.0` + `BASE_ARGS` + named pieces: `gemm` (power-of-two GEMM rows, `stages/gemm_pow2.csv`), `gemm2` (shipped table), `ua1` (attention entry), `ssm16`, `gdn` (patch), `mtpK`, `mbtN`, `image=` |
| `waitready.sh <name> <port>` | Seconds from launch to `/health` 200 |
| `gate.sh <tag> <port>` | `smoke20.py` (saved, diffed against stage B) and GSM8K first 250 at c=32 |
| `gsm8k.py <url> [n] [conc]` | GSM8K via chat, temperature 0, `chat_template_kwargs.enable_thinking=false` |
| `smoke20.py <url> [out.json]`, `--diff a b` | 20 fixed prompts, temperature 0, one at a time (deterministic); counts changed answers |
| `tpm_bench.py` | The production load generator (`/mnt/nvme/ops/tpm_bench.py` on node-146) with `TPM_URL`/`TPM_MODEL` overrides, `tpot_mean_ms`, `lit_tpm` (Little's law) and `--stagger` |
| `tpm.sh <port> <log> <steps> [secs] [warm]` | Runs `tpm_bench.py` closed-loop steps; `STAGGER=s` spreads stream starts |
| `probe.sh <port> 16 8192 1 64` | Prefill probe (`vllm bench serve`, random 8K prompts, 1 output token) |
| `measure.sh <tag> <port> [full]` | Prefill x2, c48 x2, c16 x2, one desync run each at c48 and c16, and with `full` the knee ladder x2 |
| `run.sh <tag> <port> [full]` | `gate.sh` then `measure.sh` |
| `table.py <tag>...` | Markdown rows for `../results/ablation.md` |
| `profreq.py <port> 8192 4`, `profload.sh <port> 48 50` | Prefill and c48 load traces (engine needs `--profiler-config`) |
| `analysis/qtrace.py <trace> [topn] [decode\|mixed\|prefill]` | Kernel time by category for this model (GEMM, GDN chunk/recurrent/conv, attention, quant, norm) and GPU busy |
| `analysis/{steps,stepkern,gaps,kparent,ktrace}.py` | From the Gemma recipe: per-step times, kernels in step windows, idle gaps, launching ops |
| `kernels/tune_gemm.sh`, `kernels/tune_gemm2.sh` | aiter `gemm_a8w8_blockscale_tune.py --preshuffle --libtype all` for the five shapes (power-of-two M, then exact decode/verify M) |
| `kernels/cmptune.py`, `kernels/mergetune.py`, `kernels/showtune.py` | Compare exact-M picks with the padded power-of-two row, merge rows >= 1.05x, print a tuned CSV |

## Commands for each table

```bash
cd bench; source common.sh; source engine_args.sh
# Baselines
./launch.sh qopt-a 0 8000 vllm/vllm-openai-rocm:v0.26.0 "${BASE_ARGS[@]}"
./launch.sh qopt-b 1 8001 vllm/vllm-openai-rocm:v0.30.0 "${BASE_ARGS[@]}"
# Ablation stages (one GPU each)
./stack.sh qopt-g 2 8002 gemm
./stack.sh qopt-g-ua1 3 8003 gemm ua1
./stack.sh qopt-g-ua1-ssm16 4 8004 gemm ua1 ssm16
./stack.sh qopt-s 5 8005 gemm ua1 ssm16 gdn
./stack.sh qopt-s-mtp2 6 8006 gemm ua1 ssm16 gdn mtp2
./stack.sh qopt-f 7 8007 gemm2 ua1 ssm16 gdn mtp3      # measured shipped stack (gdn is inert with MTP)
for p in 8000 8001 8002 8003 8004 8005 8006 8007; do ./waitready.sh <name> $p; done
./run.sh f 8007 full        # gate + all metrics + knee; likewise for each tag
python3 table.py A B gemm g-ua1 g-ua1-ssm16 s s-mtp2 f
# Profiles
PROF='{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"active_iterations":12}'
./stack.sh qopt-f-prof 0 8000 gemm2 ua1 ssm16 gdn mtp3 -- --profiler-config "$PROF"
python3 profreq.py 8000 8192 4; ./profload.sh 8000 48 50
python3 analysis/qtrace.py <trace> 15 decode    # and mixed, prefill
# GEMM tables
bash kernels/tune_gemm.sh; bash kernels/tune_gemm2.sh
python3 kernels/mergetune.py $WORK/tune/qwen38_bpreshuffle_tuned.csv $WORK/tune/qwen38_bpreshuffle_tuned2.csv ../qwen38_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
```

## Rules the run followed

- Gate before a number counts: GSM8K-250 >= 96.5% (a run below gets two reruns, median counts: baseline B itself
  scored 96.4, 97.6, 97.2%); `smoke20.py` diff is the strict numerics check.
- One change per engine against the previous stage; keep only if the gate passes and c48 TPM improves.
- Every metric twice except the desync runs; throughput from `lit_tpm`.
- Never edit a script that a running benchmark is executing: write to a temp file and `mv` (bash reads scripts
  incrementally; two runs were lost to this).
- `kernels/tune_gemm*.sh` and `common.sh` carry host paths from the run; edit `WORK`/`WEIGHTS` for another host.
