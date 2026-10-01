# shellcheck shell=bash disable=SC2034
# common.sh: defaults shared by the scripts in this directory. Sourced, not executed.
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(dirname "$BENCH_DIR")
WORK=$(realpath -m "${WORK:-/mnt/nvme/ops/qwen38-opt}")
# Qwen/Qwen3.8-27B-FP8 as staged by the platform (weights ref qwen3.8-27b-fp8).
WEIGHTS=$(realpath -m "${WEIGHTS:-/var/lib/openrelay/weights/836916891e86a10c107bf13013c0c351fdb16cb9f87ef7746d77ad5ddf6ac978}")
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.30.0}
# Runs the load generators (vllm bench serve, tpm_bench.py needs aiohttp); no GPU.
CLIENT_IMAGE=${CLIENT_IMAGE:-vllm/vllm-openai-rocm:v0.30.0}
MODEL=${MODEL:-qwen3.8-27b}
# ~2,500 prompt tokens for tpm_bench.py (calibrated once against this tokenizer and chat template).
WORDS=${WORDS:-1195}
export WORK WEIGHTS IMAGE CLIENT_IMAGE MODEL WORDS
# rocm-smi index -> render and card node (same PCI function; checked against /sys/class/drm on node-146).
gpu_devs() {
  case $1 in
    [0-7]) echo "/dev/dri/renderD$((128 + 8 * $1)) /dev/dri/card$((8 * $1))" ;;
    *) echo "no GPU $1" >&2; return 1 ;;
  esac
}
