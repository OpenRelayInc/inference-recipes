#!/usr/bin/env bash
# tune_gemm.sh: aiter a8w8 block-scale GEMM tuner (B-preshuffle mode, all backends) for Qwen3.8-27B's five FP8 linear
# shapes on GPUs 4..7, inside the v0.30.0 image. Output: $WORK/tune/qwen38_bpreshuffle_tuned.csv
set -euo pipefail
WORK=/mnt/nvme/ops/qwen38-opt; T=$WORK/tune; mkdir -p $T
{ echo "M,N,K"; for nk in 14336,5120 5120,6144 16384,5120 34816,5120 5120,17408; do
    for m in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do echo "$m,$nk"; done; done; } > $T/qwen38_untuned.csv
sudo chown -R 100000:100000 $T
devs=(--device /dev/kfd); for g in 4 5 6 7; do devs+=(--device /dev/dri/renderD$((128+8*g)) --device /dev/dri/card$((8*g))); done
docker rm -f qopt-tune >/dev/null 2>&1 || true
docker run -d --name qopt-tune --memory 300g --memory-swap 300g --cpus 32 --shm-size 8g --security-opt no-new-privileges \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --group-add 44 --group-add 992 "${devs[@]}" -v $T:/tune \
  --entrypoint bash vllm/vllm-openai-rocm:v0.30.0 -c "cd /usr/local/lib/python3.12/dist-packages/aiter_meta && \
  python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i /tune/qwen38_untuned.csv \
  -o /tune/qwen38_bpreshuffle_tuned.csv -o2 /tune/qwen38_bpreshuffle_all.csv --preshuffle --libtype all --mp 4 2>&1 | tee /tune/tune.log"
