# Benchmark and correctness harness

The scripts behind every number in `../results/`, generalized from the ones used in the run: paths, image,
weights, GPU and port are arguments or environment variables. Run them from the recipe directory
(`gemma-4-31b-mi355x/`) on a ROCm host with docker, `curl` and `python3`.

## Setup

`common.sh` holds the defaults every script reads. Override from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `WORK` | `./work` | Results, per-engine caches (Triton JIT, profiler traces), generated data |
| `WEIGHTS` | `$WORK/weights` | `amd/gemma-4-31B-it-MXFP4` with `google/gemma-4-31B-it-assistant` in `assistant/` (see `../README.md`, "Weights") |
| `IMAGE` | `vllm/vllm-openai-rocm:v0.30.0` | Engine image for `ablate.sh` (must be the stock image there) |
| `CLIENT_IMAGE` | `vllm/vllm-openai-rocm:v0.30.0` | Runs `vllm bench serve` as the load generator |
| `MODEL` | `gemma-4-31b-it` | Served model name |

```bash
export WORK=$PWD/work WEIGHTS=$PWD/work/weights
source bench/common.sh; source bench/engine_args.sh   # BASE_ARGS, BASE2_ARGS, BASE3_ARGS, TUNED_ARGS
TUNED_IMAGE=public.ecr.aws/o7n5w9k8/vllm-rocm-gemma4@sha256:f22210e5fbd75824e6deb8c88aaae2c449c7180f6f4c6aaa49df7775716d4cfa
```

## Scripts

Serving harness:

| Script | What it does |
|---|---|
| `launch.sh <name> <gpu> <port> <image> <weights> [args]` | Starts one TP1 engine in a container that sees only `<gpu>`. Mounts the recipe at `/recipe`. `UA_JSON`, `EXTRA_MOUNTS`, `EXTRA_ENV`, `PRE`, `MAXSEQS` add the tuned pieces to a stock image. Refuses to start if the port already answers or another running container can see the GPU |
| `engine_args.sh` | Engine argument arrays: `BASE_ARGS` (baseline), `BASE2_ARGS` (+ graphs to 256 tokens), `BASE3_ARGS` (graphs to 384, for `MAXSEQS=128`), `TUNED_ARGS` (shipped) |
| `smoke.sh <port>` | One chat prompt at temperature 0. A healthy engine names Canberra and 391 |
| `gsm8k.py <base-url> [n] [conc] [offset]` | Correctness gate: first n GSM8K test problems (250) at concurrency conc (32) |
| `probe.sh <port> <conc> [in] [out] [num]` | Closed-loop `vllm bench serve` probe: throughput, TTFT, TPOT |
| `bench.sh <name> <port> [rates]` | Open-loop Poisson sweep at the workload's shape, one run per rate; `DUR`, `SEED` |
| `bench_text.sh <name> <port> [rates]` | The same with natural-text prompts and 380 forced output tokens, for speculative-decoding work |
| `mkdata.py <tokenizer> <gsm8k.jsonl> <out.jsonl>` | Builds the natural-text prompt set for `bench_text.sh` |
| `table.py [results-dir]` | One line per `bench.sh` / `bench_text.sh` result |
| `ablate.sh [gpu] [port]` | Cumulative one-change-per-stage ablation on one GPU |
| `profreq.py <port> <input_len> <n>` | Profiles n concurrent prefill-only requests |
| `profload.sh <port> <rate> [warm_s]` | Profiles 20 s of open-loop load after a warm-up |
| `patch_gemma4_attn.py` | Day-one per-layer attention backend patch; crashed under batching, kept for the record |
| `stages/ua_v1.json`, `stages/ua_v2.json` | The attention table at ablation stages B and D (stage F onward uses `../unified_attention_gfx950.json`) |

Trace analysis (`analysis/`, plain python3, input is a torch profiler `*.pt.trace.json.gz`):

| Script | What it prints |
|---|---|
| `ktrace.py <trace> [topn]` | GPU kernel time by category (attention global/sliding, MXFP4 GEMM, quant, norm, GELU, KV cache) and the top kernels |
| `steps.py <trace>` | Per-scheduler-step GPU time, split into decode-only, prefill-only and mixed steps |
| `stepkern.py <trace> [decode\|mixed] [topn]` | Kernel time inside decode-only (or mixed) step windows, and how busy the GPU was |
| `gaps.py <trace> [mixed\|decode]` | Idle gaps between kernels inside step windows, grouped by the kernel pair around them |
| `kparent.py <trace> <kernel-substring>` | Which CPU ops (and input shapes) launch a kernel |

Kernel microbenchmarks (`kernels/`, run inside the engine image on one GPU):

| Script | What it measures |
|---|---|
| `attn_tune.py <512\|256> [10k\|16k\|mix\|dec] [quick]` | Sweeps aiter unified-attention `attn_2d` configs at vLLM's real Gemma 4 shapes (paged fp8 KV, fp8 q) |
| `attn_dec3d.py` | MTP verify-step attention (n x 3 queries, ~9.5K context): the 2D kernel aiter picks against split-KV 3D |
| `attn_dec3d_tune.py` | Sweeps split-KV 3D configs for d=512 verify steps |
| `gemm_bench.py [M ...]` | MXFP4 GEMM backends at Gemma 4 shapes: aiter assembly `gemm_a4w4` against aiter Triton `gemm_afp4wfp4` |
| `hipblaslt_fp4.py` | Whether `torch._scaled_mm` (hipBLASLt) runs MXFP4 with e8m0 1x32 scales on gfx950, and how fast |
| `fuse_bench.py` | aiter `act_mul_and_mxfp4_quant` (GELU-tanh times up, fused with the MXFP4 quant) feeding the assembly GEMM against vLLM's unfused path |
| `fuse_bench2.py` | aiter `fused_rms_mxfp4_quant` feeding the assembly GEMM against the unfused path |
| `attnbench.py` | Day-one isolated attention kernels at the 10K prefill shape (SDPA for head_dim 512, CK flash attention for head_dim 256 sliding window) |
| `actq_tune.py` | Block-size sweep of the fused GELU+MXFP4 quant kernel at the down_proj input |
| `g4_a4w4_untuned.csv` | The 66 (M, N, K) shapes given to aiter's a4w4 GEMM tuner |

## The method these scripts support

- **Gate before any throughput number counts:** `smoke.sh` gives a sane answer and `gsm8k.py <url> 250 32`
  scores at or above 96.5%. Anything that changes numerics (kernels, tiles, quantization, fusions) must pass.
- **Pure prefill:** `probe.sh <port> 16 10000 1 64`, run twice; report the second (the first can include JIT
  warm-up).
- **Knee:** `bench.sh` across rates; the SLO-safe rate is the highest with p95 TTFT under 4 s. For a final
  comparison use `DUR=300` and two seeds, and count a rate only if both seeds pass.
- **One change at a time**, and profile before choosing the next change.

## Commands for each table in `../results/`

### Knee (`knee.md`, `knee.csv`)

Three engines on three GPUs, measured in the same session. In the run the two tuned engines were the stock
image with the tuned files mounted and both patches applied at start (the image did not exist yet); the
tuned image bakes in byte-identical files, so either form reproduces them.

```bash
KC='{"ir_op_priority":{"rms_norm":["native"]}}'
MC=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/gemma4_a4w4_blockscale_tuned_gemm.csv
bench/launch.sh g4-base 0 8000 vllm/vllm-openai-rocm:v0.30.0 "$WEIGHTS" "${BASE_ARGS[@]}"
UA_JSON=unified_attention_gfx950.json EXTRA_MOUNTS="$PWD/gemma4_a4w4_blockscale_tuned_gemm.csv:$MC:ro" \
  PRE="python3 /recipe/patch_g4_fuse.py && python3 /recipe/patch_aiter_ua3d.py" EXTRA_ENV=G4_FUSE_GELU=1 \
  bench/launch.sh g4-tuned16k 1 8001 vllm/vllm-openai-rocm:v0.30.0 "$WEIGHTS" "${BASE2_ARGS[@]}" --kernel-config "$KC"
bench/launch.sh g4-tuned 2 8002 "$TUNED_IMAGE" "$WEIGHTS" "${TUNED_ARGS[@]}"

# wait for each engine: until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
for p in 8000 8001 8002; do bench/smoke.sh $p; python3 bench/gsm8k.py http://127.0.0.1:$p 250 32; done

for s in 7 8; do
  SEED=$s DUR=300 bench/bench.sh base 8000 1.75 2.0 2.25 2.5 &
  SEED=$s DUR=300 bench/bench.sh tuned16k 8001 2.75 3.0 3.25 3.5 &
  SEED=$s DUR=300 bench/bench.sh tuned 8002 2.75 3.0 3.25 3.5 &
  wait
done
python3 bench/table.py
```

### Ablation (`ablation.md`, `ablation.csv`)

```bash
bench/ablate.sh 0 8007 | tee "$WORK/results/ablation.log"
```

Stage A is `BASE_ARGS` on the stock image; each later stage adds one piece (see the stage list in the script).
Prefill tok/s is the request rate of probe runs 2 and 3 times 10,000.

### Headline prefill probe (`README.md`, first row)

```bash
bench/probe.sh 8000 16 10000 1 64; bench/probe.sh 8000 16 10000 1 64   # baseline: 2.78, 2.77 req/s
bench/probe.sh 8002 16 10000 1 64; bench/probe.sh 8002 16 10000 1 64   # tuned: 4.19 req/s on the second run
```

### Exploration (`exploration.md`)

Single-seed sweeps: `bench/bench.sh <tag> <port> <rates>` with the defaults (`DUR=150`, `SEED=7`), rates as in
each table. The engine configs map to `launch.sh` settings as follows (all on the stock image, `GEMM` meaning
`EXTRA_MOUNTS="$PWD/gemma4_a4w4_blockscale_tuned_gemm.csv:$MC:ro"`, `P3` meaning
`PRE="python3 /recipe/patch_g4_fuse.py && python3 /recipe/patch_aiter_ua3d.py" EXTRA_ENV=G4_FUSE_GELU=1`):

| Tag | Settings |
|---|---|
| base | `BASE_ARGS` |
| ua1 | `UA_JSON=bench/stages/ua_v1.json`, `BASE_ARGS` |
| gemm1 | ua1 + `GEMM` |
| ua2 | `UA_JSON=bench/stages/ua_v2.json`, `BASE_ARGS` |
| fuse1 | `ua_v2.json` + `GEMM` + `PRE="python3 /recipe/patch_g4_fuse.py" EXTRA_ENV=G4_FUSE_GELU=1`, `BASE_ARGS` |
| cg1 | `ua_v2.json` + `GEMM`, `BASE2_ARGS` |
| cg2 | cg1 + `PRE="python3 /recipe/patch_g4_fuse.py" EXTRA_ENV=G4_FUSE_GELU=1` |
| cg3 | `UA_JSON=unified_attention_gfx950.json` + `GEMM` + `P3`, `BASE2_ARGS` |
| mtp3 | cg3 with `"num_speculative_tokens":3` in `--speculative-config` |
| qkf1 | cg3 + `--compilation-config` with `"pass_config":{"enable_qk_norm_rope_fusion":true}` |
| qkf2 | cg3 + `"use_inductor_graph_partition":true` and `"pass_config":{"fuse_qk_norm_rope_kvcache":true,"rope_kvcache_fusion_max_token_num":16384}` |
| nat1 | cg3 + `--kernel-config '{"ir_op_priority":{"rms_norm":["native"],"fused_add_rms_norm":["native"]}}'` |
| nat2 | cg3 + `--kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}'` (ablation stage G) |
| pw1 | nat2 with piecewise capture sizes every 512 tokens up to 16384 |
| s128 | cg3 with `MAXSEQS=128` and `BASE3_ARGS` (then also `--max-num-batched-tokens 8192`) |
| b8k | nat2 + `--max-num-batched-tokens 8192` (the shipped config) |

For qkf1, qkf2 and pw1, merge the extra keys into the same `--compilation-config` JSON that carries the capture
sizes (vLLM keeps only the last `--compilation-config` flag).

Natural-text bench (MTP k=2 against k=3):

```bash
mkdir -p "$WORK/data"
curl -L -o "$WORK/data/gsm8k_test.jsonl" \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
docker run --rm -v "$WEIGHTS":/tok:ro -v "$WORK/data":/data -v "$PWD/bench":/bench:ro \
  --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /bench/mkdata.py /tok /data/gsm8k_test.jsonl /data/text_bench.jsonl
bench/bench_text.sh cg3 <port> 2.5 3.0 3.5
bench/bench_text.sh mtp3 <port> 2.5 3.0 3.5
```

Acceptance rates come from the engine's `/metrics` (`vllm:spec_decode_num_accepted_tokens` and related
counters) and its log.

### Profiles (`profiles.md`)

Add the profiler to the engine args; traces land in `$WORK/cache/<engine-name>/prof/`:

```bash
PROF='{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"active_iterations":3}'
bench/launch.sh g4-prof 3 8005 vllm/vllm-openai-rocm:v0.30.0 "$WEIGHTS" "${BASE_ARGS[@]}" --profiler-config "$PROF"
python3 bench/profreq.py 8005 10000 3      # prefill, 10K; then: profreq.py 8005 16000 2
python3 bench/analysis/ktrace.py "$WORK"/cache/g4-prof/prof/<trace>.pt.trace.json.gz
bench/profload.sh 8005 2.5 60              # decode steps under load
python3 bench/analysis/steps.py <trace>; python3 bench/analysis/stepkern.py <trace> decode
python3 bench/analysis/gaps.py <trace> mixed; python3 bench/analysis/kparent.py <trace> copy
```

The load profiles in the run used a capture window sized by `active_iterations`; its exact value for those
runs was not recorded.

### Kernel microbenchmarks (`kernels.md`)

```bash
UAD=/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json
k() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0 \
        -v "$PWD/bench/kernels":/k:ro "$@"; }
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/attn_tune.py 512 10k quick
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/attn_tune.py 512 10k      # also 256 10k, 512 dec, 512 16k, 512 mix
k -v "$PWD/bench/stages/ua_v2.json:$UAD:ro" --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/attn_dec3d.py
k -v "$PWD/bench/stages/ua_v2.json:$UAD:ro" --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/attn_dec3d_tune.py
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/gemm_bench.py 16384
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/hipblaslt_fp4.py
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/fuse_bench2.py
k --entrypoint python3 vllm/vllm-openai-rocm:v0.30.0 /k/actq_tune.py
```

The GEMM table came from aiter's tuner, run from a checkout of `ROCm/aiter` at tag `v0.1.21.post2` inside the
engine image: `python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py -i g4_a4w4_untuned.csv --compare`
(output path: `-o`, per the tuner's README). The rows kept are the 44 that beat the untuned default by 3% or more;
that filter was applied by hand.

The in-engine split-KV numbers (81 to 641 us) were measured through aiter's own kernel selection with the
patch and the shipped table in place; that exact invocation was not recorded.

## Not preserved from the run

- The hand filter over the GEMM tuner output (rows at least 3% faster than the default were kept).

## Differences from the scripts used in the run

- Paths, image, weights, GPU, port and served model name are parameters (`common.sh`, arguments).
- `launch.sh` refused GPUs outside the four set aside for the run on a shared host; here it refuses a GPU that
  any other running container can see, and a port that already answers.
- `bench.sh` is the run's seeded sweep (`SEED`, `DUR`); with the defaults it is the run's original single-seed
  sweep. Its header now states the output length range the command actually samples (76 to 684 tokens; the
  old comment said 190 to 570). The random dataset does not set `--ignore-eos`, so outputs can end earlier.
- Every client loads the tokenizer from `$WEIGHTS`. In the run, `probe.sh` and the first `bench.sh` loaded it
  from a different Gemma 4 checkpoint's directory.
- `launch.sh` mounts this recipe at `/recipe` (read-only) so `PRE` can run the patches; the run mounted its work
  directory instead. The patches are the shipped versions, which add a TP > 1 guard and a positive check of the
  `scaleM_pad` edit to the versions used in the run; at TP1 they do the same thing.
- The SGLang launch mode of the first launcher is dropped (SGLang could not load the checkpoint).
