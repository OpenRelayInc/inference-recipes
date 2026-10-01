#!/usr/bin/env bash
# tune_gemm2.sh: second aiter tuner pass for decode-sized M between the power-of-two buckets of the first pass
# (24..192: 48 concurrent decodes, and MTP verify batches of 48 x 2 or 48 x 3 tokens). GPUs 4 and 6.
set -euo pipefail
WORK=/mnt/nvme/ops/qwen38-opt; T=$WORK/tune; sudo mkdir -p $T; sudo chown -R ubuntu $T
{ echo "M,N,K"; for nk in 14336,5120 5120,6144 16384,5120 34816,5120 5120,17408; do
    for m in 24 40 48 56 80 96 112 144 160 192; do echo "$m,$nk"; done; done; } > $T/qwen38_untuned2.csv
sudo chown -R 100000:100000 $T
devs=(--device /dev/kfd); for g in 4 6; do devs+=(--device /dev/dri/renderD$((128+8*g)) --device /dev/dri/card$((8*g))); done
docker rm -f qopt-tune2 >/dev/null 2>&1 || true
docker run -d --name qopt-tune2 --memory 300g --memory-swap 300g --cpus 32 --shm-size 8g --security-opt no-new-privileges \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --group-add 44 --group-add 992 "${devs[@]}" -v $T:/tune \
  --entrypoint bash vllm/vllm-openai-rocm:v0.30.0 -c "cd /usr/local/lib/python3.12/dist-packages/aiter_meta && \
  python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i /tune/qwen38_untuned2.csv \
  -o /tune/qwen38_bpreshuffle_tuned2.csv -o2 /tune/qwen38_bpreshuffle_all2.csv --preshuffle --libtype all --mp 2 2>&1 | tee /tune/tune2.log"
