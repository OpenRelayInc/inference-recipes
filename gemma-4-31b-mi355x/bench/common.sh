# shellcheck shell=bash disable=SC2034
# common.sh: defaults shared by the scripts in this directory. Sourced, not executed.
# Override any value from the environment, e.g. WORK=/data/g4 WEIGHTS=/models/g4 ./probe.sh 8000 16
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(dirname "$BENCH_DIR")
# Results, per-engine caches (Triton JIT, torch profiler traces) and generated data. Absolute, for docker -v.
WORK=$(realpath -m "${WORK:-$PWD/work}")
# amd/gemma-4-31B-it-MXFP4 at the root, google/gemma-4-31B-it-assistant in assistant/ (README.md, "Weights").
WEIGHTS=$(realpath -m "${WEIGHTS:-$WORK/weights}")
# Engine image. The stock image is the baseline; the tuned pieces are mounted or patched in at start.
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.30.0}
# Image that runs the load generator (`vllm bench serve`); any vLLM image works.
CLIENT_IMAGE=${CLIENT_IMAGE:-vllm/vllm-openai-rocm:v0.30.0}
# Served model name, passed to the engine as --served-model-name and used by every client.
MODEL=${MODEL:-gemma-4-31b-it}
# Exported so scripts called from other scripts (ablate.sh runs launch.sh, smoke.sh, probe.sh) resolve the same paths.
export WORK WEIGHTS IMAGE CLIENT_IMAGE MODEL
