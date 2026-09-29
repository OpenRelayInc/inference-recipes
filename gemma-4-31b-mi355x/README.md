# Gemma 4 31B on AMD MI355X

A vLLM ROCm engine for Gemma 4 31B (AMD's Quark MXFP4 W4A4 checkpoint, MTP speculative decoding) on AMD
Instinct MI355X, and the harness that measured it. It is stock `vllm/vllm-openai-rocm:v0.30.0` plus two tuned
config tables, two small source patches, and three engine flags.

We moved a production Gemma 4 31B deployment from NVIDIA B200 to MI355X, then ran an agent-driven tuning
loop on four MI355X GPUs for about three hours. On a long-prompt production workload (~9.5K tokens in, ~380
out) each GPU now serves 1.5x the requests under a 4 s p95 time to first token, with GSM8K unchanged. This
directory is what serves that traffic.

## Results

One MI355X, TP1, same host and harness before and after.

**Baseline:** our first working MI355X config, stock `vllm/vllm-openai-rocm:v0.30.0` with the production
arguments (`BASE_ARGS` in `bench/engine_args.sh`). The 1.5x is against that config, not against B200; we
never ran this harness on B200.

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Requests/s per GPU with p95 TTFT < 4 s (300 s per rate, both of two seeds) | 2.0 | 3.0 (3.25 borderline) | **1.5x** |
| Prefill tokens/s per GPU (16 concurrent 10K-token prompts) | 27.8K | 41.9K | **1.51x** |
| Median time per output token at the highest SLO-safe rate | 40 to 43 ms | 22 ms | **-47%** |
| Saturated input tokens/s, open loop | 21.7K | 32.2K | 1.48x |
| GSM8K, first 250 test problems | 98.8% | 98.8% | unchanged |

Where each gain came from, one change at a time on one GPU:

| Stage | Prefill tok/s | Closed-loop req/s (32 users, 9.5K in, 380 out) | Mean TPOT ms |
|---|---|---|---|
| Baseline | 27.9K | 2.08 | 32.9 |
| + attention tile entries (`unified_attention_gfx950.json`) | 37.3K | 2.32 | 27.9 |
| + tuned MXFP4 GEMM rows (`gemma4_a4w4_blockscale_tuned_gemm.csv`) | 38.7K | 2.42 | 28.7 |
| + CUDA-graph capture sizes to 256 tokens | 38.8K | 2.88 | 22.6 |
| + fused GELU, up and MXFP4 quant (`patch_g4_fuse.py`) | 39.4K | 2.94 | 22.6 |
| + split-KV decode attention (`patch_aiter_ua3d.py`) | 39.5K | 2.98 | 22.4 |
| + native RMSNorm (`--kernel-config`) | 41.7K | 3.05 | 21.0 |

The graph-capture stage also added the head_dim 512 verify-step tile from the attention table, which was
neutral at the knee on its own. The shipped config adds an 8192-token prefill budget to the last row, which
lowers time per output token at no cost to prefill.

In production, at 80% of the traffic, p95 time to first token held at or under 0.75 s (the same as
B200) with 14 ms between tokens and no errors, and the full 1,319-problem GSM8K scored 97.3% (97.1% before
tuning).

All tables, including what was tried and discarded: [results/](results/README.md).

## What is here

| Path | What it is |
|---|---|
| `Dockerfile` | `vllm/vllm-openai-rocm:v0.30.0` plus the four files below |
| `unified_attention_gfx950.json` | aiter's gfx950 unified-attention config table with Gemma 4 entries (replaces the image's `DEFAULT.json`) |
| `gemma4_a4w4_blockscale_tuned_gemm.csv` | 44 tuned MXFP4 GEMM rows for Gemma 4 31B's shapes (merged by aiter from `model_configs/`) |
| `patch_g4_fuse.py` | Fuses GELU(tanh)*up and the MXFP4 activation quant ahead of the down projection; fixes a mid-serving JIT storm in aiter |
| `patch_aiter_ua3d.py` | Lets aiter use split-KV attention for head_dim 512 on decode and MTP verify steps |
| `bench/` | Launcher, correctness gate, load generators, profiling and trace analysis, kernel microbenchmarks ([bench/README.md](bench/README.md)) |
| `results/` | Every measured table, as markdown (and CSV for the two main ones) |

## Hardware and software

- AMD Instinct MI355X (gfx950), one GPU per engine (TP1).
- `vllm/vllm-openai-rocm:v0.30.0`: ROCm 7.2.3, aiter `v0.1.21.post2`. Both tables are keyed to this aiter
  version's kernels.
- Tuned image: `public.ecr.aws/o7n5w9k8/vllm-rocm-gemma4:v0.30.0-g4mi355`, digest
  `sha256:f22210e5fbd75824e6deb8c88aaae2c449c7180f6f4c6aaa49df7775716d4cfa`, built from the `Dockerfile` in this
  directory. The four files it adds are byte-identical to the ones here, except that the GEMM CSV in this
  repository has LF line endings (the image's copy has CRLF; both parse to the same table).

## Weights

Two public Hugging Face repositories, both Apache-2.0 and not gated. The engine expects the main checkpoint at
the root of one directory and the MTP drafter in its `assistant/` subdirectory:

```bash
pip install -U huggingface_hub
export WEIGHTS=$PWD/work/weights
hf download amd/gemma-4-31B-it-MXFP4 --revision 0606edf12b2a926dc998b2f7684fe395ee6d2a69 --local-dir "$WEIGHTS"
hf download google/gemma-4-31B-it-assistant --revision 627c5ec1458b9086b841a91e0512fd31fd2fbbf1 \
  --local-dir "$WEIGHTS/assistant"
```

- `amd/gemma-4-31B-it-MXFP4`: AMD Quark MXFP4 (FP4 weights and activations, per-group MX scales); its
  `model.safetensors` is 19.5 GB. `lm_head` and the vision tower stay unquantized.
- `google/gemma-4-31B-it-assistant`: Google's MTP draft model for Gemma 4 31B (`Gemma4AssistantForCausalLM`,
  0.47B parameters, BF16, 0.94 GB). The same drafter served the B200 deployment next to an NVFP4 checkpoint;
  it runs unchanged next to the MXFP4 one.
- The weight and tokenizer files we served are byte-identical (sha256) to these revisions:
  `model.safetensors` `d3f056b4c0657b5452971b39e326d86b65e6e0058632967478aba1edf24c4f51` and
  `assistant/model.safetensors` `9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da`.

## Run

```bash
docker run -d --name gemma4 \
  --network host --ipc=host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 -e VLLM_ROCM_USE_AITER=1 \
  -v "$WEIGHTS":/weights:ro -v "$PWD/work/cache":/root/.cache \
  public.ecr.aws/o7n5w9k8/vllm-rocm-gemma4@sha256:f22210e5fbd75824e6deb8c88aaae2c449c7180f6f4c6aaa49df7775716d4cfa \
  /weights --served-model-name gemma-4-31b-it --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 64 --enable-prompt-tokens-details --enable-prefix-caching \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}' \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --speculative-config '{"method":"mtp","model":"/weights/assistant","num_speculative_tokens":2}' \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,224,256]}' \
  --kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}'

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31b-it","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":40}'
```

The image's entrypoint is `vllm serve`, and it sets `G4_FUSE_GELU=1` and `TRITON_CACHE_DIR=/root/.cache/triton`.
Keep `/root/.cache` on a persistent volume so a restarted engine does not recompile its Triton kernels while
serving. The same launch through the harness: `bench/launch.sh gemma4 0 8000 <image> "$WEIGHTS" "${TUNED_ARGS[@]}"`
after `source bench/engine_args.sh`.

**Stock image, no build.** Mount the table and the CSV over the image's copies, apply both patches before
`vllm serve`, and set `G4_FUSE_GELU=1`. `bench/launch.sh` does this with `UA_JSON`, `EXTRA_MOUNTS`, `PRE` and
`EXTRA_ENV` (the tuned engines in the benchmark runs were launched this way; commands in `bench/README.md`).

**Flags only.** Three of the changes need nothing but flags on `vllm/vllm-openai-rocm:v0.30.0`:

```bash
--max-num-batched-tokens 8192
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,224,256]}'
--kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}'
```

## Build

```bash
docker build -t vllm-rocm-gemma4:v0.30.0-g4mi355 .
```

Any docker host works; the build needs no GPU. Both patches locate vLLM and aiter without importing them
(aiter probes the GPU on import) and fail the build if the source they edit has moved. When the base image
moves past v0.30.0, re-check both patches (the build fails if an anchor is missing) and re-tune both tables:
they are keyed to aiter `v0.1.21.post2`.

## Reproduce the benchmarks

`bench/README.md` has the exact commands for every table. The short version, on a host with three free
MI355X GPUs:

```bash
export WORK=$PWD/work WEIGHTS=$PWD/work/weights
source bench/common.sh; source bench/engine_args.sh
TUNED_IMAGE=public.ecr.aws/o7n5w9k8/vllm-rocm-gemma4@sha256:f22210e5fbd75824e6deb8c88aaae2c449c7180f6f4c6aaa49df7775716d4cfa
bench/launch.sh g4-base 0 8000 vllm/vllm-openai-rocm:v0.30.0 "$WEIGHTS" "${BASE_ARGS[@]}"
bench/launch.sh g4-tuned 1 8001 "$TUNED_IMAGE" "$WEIGHTS" "${TUNED_ARGS[@]}"
# when both answer on /v1/models:
bench/smoke.sh 8001; python3 bench/gsm8k.py http://127.0.0.1:8001 250 32   # the correctness gate
bench/probe.sh 8001 16 10000 1 64; bench/probe.sh 8001 16 10000 1 64         # prefill, second run counts
for s in 7 8; do SEED=$s DUR=300 bench/bench.sh base 8000 1.75 2.0 2.25 2.5; done
for s in 7 8; do SEED=$s DUR=300 bench/bench.sh tuned 8001 2.75 3.0 3.25 3.5; done
python3 bench/table.py
bench/ablate.sh 2 8007                                                        # the per-change table
```

`bench/launch.sh` refuses to start an engine if something already answers on its port or another running
container can see its GPU. In the run, one launch failed on a port still held by an older engine, and the next
round of benchmarks quietly measured the older engine; those numbers were thrown out and this check added.

## What each piece does

**Baseline arguments (the first working config).** On ROCm, vLLM runs Gemma 4 on its Triton attention
backend by default (the layers mix head_dim 256 and 512); that spent 67% of prefill in attention, most of it
in the 10 global layers. `--attention-backend ROCM_AITER_UNIFIED_ATTN` runs aiter's unified attention on
every layer and was 1.8x faster.
`--kv-cache-dtype fp8` also makes vLLM quantize the query to fp8, so aiter reads its `DT_fp8_fp8` table
entries. MTP with 2 draft tokens beat 1 and none. `--limit-mm-per-prompt` turns off image, audio and video
inputs, and `--hf-overrides` was carried over unchanged from the B200 deployment.

**`unified_attention_gfx950.json` (a third more prefill).** aiter picks attention tile sizes from a
per-architecture JSON table, and the gfx950 table in aiter `v0.1.21.post2` has no prefill entry for head_dim 256 or more. So
Gemma 4's 10 global layers (head_dim 512) and 50 sliding-window layers (head_dim 256, window 1024) both ran a
decode-sized tile of 16 query rows. The global layers put 8 query heads on each KV head, so that is 2 tokens
per program, with every K/V tile re-read for every 2 tokens. Against aiter's stock table, this file changes
five entries:

| Section | Key | Setting | Why |
|---|---|---|---|
| `attn_2d` (new) | `D_GEQ_512.Q_GEQ_256.DT_fp8_fp8` | BLOCK_M 128, TILE 64, 4 warps, 1 stage, waves_per_eu 1 | Global-layer prefill: 10.6 to 3.9 ms per step |
| `attn_2d` (new) | `D_GEQ_256.Q_GEQ_256.SW.DT_fp8_fp8` | BLOCK_M 128, TILE 64, 4 warps, 1 stage, waves_per_eu 2 | Sliding-layer prefill: 1.37 to 0.59 ms per step |
| `attn_2d` (new) | `D_GEQ_512.Q_LEQ_16.DT_fp8_fp8` | BLOCK_M 32, TILE 64, 4 warps, 2 stages, waves_per_eu 1 | Global-layer MTP verify steps: 0.62 to 0.42 ms |
| `attn_3d` (changed) | `D_GEQ_512` | BLOCK_M 32, 2 warps, 1 stage, waves_per_eu 1 (was 16, 4, 1, 2) | Split-KV decode, used with `patch_aiter_ua3d.py` |
| `kv_split` (changed) | `D_GEQ_512.DT_fp8_fp8` | MIN_SEGMENTS 4, SEGMENTS_PER_CU 4, TILE 64 (was 16, 8, TILE 32 to 64) | Same |

The prefill entries are keyed on 256 or more query tokens, so decode keeps its own tiles. In the engine, prefill
attention time fell from 482 to 182 ms for three 10K-token prompts, and prefill throughput rose by a third.

**`gemma4_a4w4_blockscale_tuned_gemm.csv` (+4% prefill).** The engine log prints "not found tuned config" for
every Gemma 4 MXFP4 GEMM shape, so each ran aiter's untuned default. These are aiter's tuner picks for the six
(N, K) shapes across M from 16 to 16384, keeping the 44 rows that beat the default by 3% or more. The gains are on
the N=5376 shapes (the output and down projections), 1.2x to 1.6x at small and large M; the large-N shapes
were already on the best kernel. aiter merges any CSV under `aiter/configs/model_configs/` at start.

**CUDA-graph capture sizes to 256 tokens (+19% closed-loop requests).** Under load, a decode step took 57.7 ms
of which 21.3 ms was kernels: the rest was the CPU launching kernels one by one. vLLM captured CUDA graphs (HIP
graphs on ROCm) only up to 64 tokens, and with two draft tokens each decoding request is 3 tokens, so any decode
batch over 21 requests ran without a graph. Capturing to 256 tokens covers 64 sequences, costs 7 s of startup
and 3.1 GiB, and made decode steps fully GPU-bound.

**`patch_g4_fuse.py` (+1.5% prefill).** Gemma 4's MLP computes GELU(tanh)(gate) * up, quantizes it to MXFP4, and
feeds the down projection. The patch registers a custom op that does the activation and quantization in one
aiter Triton kernel (`act_mul_and_mxfp4_quant`) and calls the assembly GEMM directly: 1.516 to 1.377 ms per layer
at 16K tokens. It is on only with `G4_FUSE_GELU=1`, only for quantized down projections (the BF16 drafter keeps
its path), and only at TP1, because it bypasses the row-parallel all-reduce. It also fixes a compile storm in
aiter: `_act_mul_and_dynamic_mxfp4_quant_kernel` declared its padded row count `scaleM_pad` as `tl.constexpr`,
so every new batch size (rounded to 256) compiled a new Triton kernel mid-serving, about 0.2 s each and 64 or
more variants up to 16K tokens. It only feeds a mask comparison; the patch makes it a runtime argument and
asserts that it did.

**`patch_aiter_ua3d.py`.** aiter forces its 2D attention kernel for head_dim 512 or more whenever a step has
more than one query token. MTP verify steps have 3 per sequence, so a 14-request step launched 56 programs on
256 compute units, each walking ~9.5K tokens of context. The patch keeps the 2D kernel for prefill (more than 16
query tokens) and lets aiter pick split-KV 3D otherwise: 361 to 85 us at 8 sequences, 875 to 650 us at 64.

**Native RMSNorm (+6% prefill).** `--kernel-config '{"ir_op_priority":{"rms_norm":["native"]}}'` swaps aiter's
RMSNorm custom op for the native implementation. The custom op is opaque to `torch.compile`, which forced
copies of the q/k head views and separate norm and RoPE kernels; with the native op Inductor fuses the head
split, per-head norm and RoPE into one kernel.

**`--max-num-batched-tokens 8192`.** Halving the prefill chunk lowers time per output token at the knee
(TPOT p50 at 3.0 req/s: 24 to 22 ms) and costs no prefill throughput.

## How the tuning was run

An agent got four GPUs of an 8-GPU MI355X host (the other four served unrelated production traffic), a brief,
and an append-only journal. A second agent read aiter's and vLLM's source and pull requests in parallel and
ranked levers; its top pick was the attention table. The rules:

- Profile first, then chase the biggest remaining slice. Stop a line of work when the profile says its ceiling
  is small.
- One change at a time: hypothesis, smallest experiment that can refute it, measure, keep or discard.
- No throughput number counts until the config passes the gate: a sane smoke answer and GSM8K (250 problems,
  c=32) at or above 96.5%.
- Same harness for every config: pure prefill with 16 concurrent 10K prompts (twice, the second run counts),
  open-loop sweeps for the knee, and a torch profile for anything that moved.
- Final comparison: 300 s per rate, two seeds, a rate counts only if both pass. Single 150 s runs at the cliff
  proved too noisy.

Two mistakes were caught and corrected during the run: the port collision described above, and the fusion's
compile storm, which made every config with the fusion look worse at the knee until it was fixed.

## What did not work

- **SGLang** (`lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260928`): cannot load this checkpoint (a shape
  assert in `load_merged_column_weight`), after also needing a processor config the checkpoint lacks.
- **Other MXFP4 GEMM backends:** aiter's Triton `gemm_afp4wfp4` (1.7 to 1.9 PFLOP/s) and hipBLASLt through
  `torch._scaled_mm` (1.7 to 2.4 PFLOP/s) against 4.1 to 4.7 for aiter's assembly kernels.
- **Tuning the large-N GEMM shapes** (qkv, gate_up): already on the best kernel.
- **Fused RMSNorm + MXFP4 quant** (aiter `fused_rms_mxfp4_quant`): 0 to 3% at prefill, slower at decode.
- **Block sizes of the fused GELU+quant kernel:** the default is already best.
- **vLLM v0.30's QK-norm, RoPE and KV-cache fusion passes:** they do not match Gemma 4 (head_dim 512 is
  unsupported, and its V norm and K = V layout miss the pattern), so they replace nothing.
- **CUDA graphs up to 16K tokens:** no gain (prefill is GPU-bound), 14 GiB of memory.
- **128 concurrent sequences instead of 64:** 1 to 3% more saturated throughput for twice the time per token.
- **A third draft token:** 7 to 10% lower time per token below the knee on natural text, same knee. Optional.
- **A decode tile for the head_dim 256 sliding layers:** 1.11x in isolation, not added.
- **Before the run:** an FP8 block-quantized checkpoint (5.2K prefill tok/s: untuned GEMMs), vLLM's default
  Triton attention, and CK flash attention on the sliding layers with aiter on the global ones (crashed under
  batching). With the tuned table, the sliding layers already run at CK flash-attention speed.

## Caveats

- **The random-token benchmark flatters MTP.** `bench.sh` uses random-token prompts, and the model's
  continuations of them are very predictable: 97 to 98% acceptance per draft position, against 0.87 to 0.90 and
  0.79 to 0.80 on natural text. Kernel comparisons stay valid (every config sees the same prompts), but use
  `bench_text.sh` for anything that touches speculative decoding.
- **One workload shape.** Every number is for ~9.5K tokens in and ~380 out (96% prefill) with a p95 TTFT SLO.
  Short prompts or long outputs shift the balance toward decode; this harness did not measure those shapes.
- **One host.** All comparisons ran on one GPU host whose other four GPUs were busy with unrelated work. The
  same baseline measured 28.5K prefill tok/s on another MI355X host and 27.8K on this one.
- **Version-bound.** The tables and patches are for `vllm/vllm-openai-rocm:v0.30.0` and aiter `v0.1.21.post2`.
- **Newer vLLM: turn the assembly MXFP4 GEMM back on.** vLLM `v0.30.0` always used AITER's assembly MXFP4
  GEMM on gfx950. Later builds make it opt-in with `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`, and without it the
  linears run AITER's Triton FP4 GEMM. On a 2026-09-28 nightly (`0.30.1rc1`) this model's prefill throughput
  (16 concurrent 10K-token prompts, one MI355X) was 18.9K tok/s without the variable and 27.7K with it.
- **TP1 only for the fusion.** `patch_g4_fuse.py` switches itself off at TP > 1.
- **Not much left.** MXFP4 GEMMs are now 61% of prefill time and run at 55 to 61% of the MXFP4 peak at the
  clocks the GPU holds under load. Better attention and elementwise kernels are worth about 1.2 to 1.3x more
  ([results/README.md](results/README.md), "Distance from the hardware ceiling").

## Upstream

Everything here that belongs in a library is going upstream, re-measured against current AITER and vLLM `main`
on an MI355X:

| Change | Pull request | On AITER `main` |
|---|---|---|
| gfx950 attention prefill tile for head_dim 512 | [ROCm/aiter#5926](https://github.com/ROCm/aiter/pull/5926) | 2.71x to 2.81x on the global-layer prefill kernel |
| `scaleM_pad` as a runtime argument in the act_mul + MXFP4 quant kernels | [ROCm/aiter#5927](https://github.com/ROCm/aiter/pull/5927) | 12 JIT compiles across 14 batch sizes become 4; outputs bit-identical |
| Gemma 4 31B MXFP4 GEMM rows for gfx950 | [ROCm/aiter#5928](https://github.com/ROCm/aiter/pull/5928) | tuned rows 1.03x to 1.82x faster, 1.078x over all 66 shapes |
| Split-KV attention for head_dim 512 on short-query steps | [ROCm/aiter#5929](https://github.com/ROCm/aiter/pull/5929) | MTP verify steps 1.45x to 3.78x faster; pure decode up to 1.10x |
| Fused act_mul + MXFP4 quant inside vLLM's AITER MXFP4 GEMM (a compile pass, any model, any TP) | [vllm-project/vllm#59153](https://github.com/vllm-project/vllm/pull/59153) | +1.8% prefill on the assembly GEMM path, GSM8K unchanged |

One change from this recipe is not going upstream: on AITER `main`, the 256-dim sliding-window layers now run
AITER's gfx950 Gluon kernel, which is already faster than the Triton kernel with the tuned entry this recipe
ships (0.55 ms against 0.61 ms at the 10K prefill shape).

