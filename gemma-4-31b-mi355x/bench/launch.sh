#!/usr/bin/env bash
# launch.sh <name> <gpu> <port> <image> <weights-dir> [engine args...]
# One TP1 Gemma 4 31B engine on one MI355X, detached, in a container named <name> that sees only GPU <gpu>.
#   UA_JSON=<file>        bind-mounts a replacement gfx950 unified-attention DEFAULT.json
#   EXTRA_MOUNTS="h:c .." adds bind mounts (the tuned GEMM CSV goes in this way, see ablate.sh)
#   EXTRA_ENV="K=V .."    adds environment variables (G4_FUSE_GELU=1 turns on the fused MLP path)
#   PRE="cmd"             runs a shell command before vllm serve; this recipe directory is mounted at
#                         /recipe, so PRE="python3 /recipe/patch_aiter_ua3d.py" applies a patch at start
#   MAXSEQS               --max-num-seqs (default 64)
# Refuses to start if something already answers on <port>, or if another running container can see <gpu>.
# Without that check a failed launch leaves the old engine on the port, and every benchmark after it
# silently measures the old engine.
set -euo pipefail
source "$(dirname "$0")/common.sh"
name=$1 gpu=$2 port=$3 image=$4 weights=$(realpath "$5"); shift 5
docker rm -f "$name" >/dev/null 2>&1 || true
if curl -s -o /dev/null "localhost:$port/v1/models"; then echo "port $port busy" >&2; exit 1; fi
for c in $(docker ps --format "{{.Names}}"); do
  vis=$(docker inspect "$c" --format "{{range .Config.Env}}{{println .}}{{end}}" \
    | sed -nE "s/^(HIP|ROCR|CUDA)_VISIBLE_DEVICES=//p" | head -n1)
  if [[ -n "$vis" ]]; then
    if [[ ",$vis," == *",$gpu,"* ]]; then echo "GPU $gpu held by $c ($vis)" >&2; exit 1; fi
  elif docker inspect "$c" --format "{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}" | grep -qx /dev/kfd; then
    echo "GPU $gpu may be in use: $c maps /dev/kfd with no *_VISIBLE_DEVICES, so it sees every GPU" >&2; exit 1
  fi
done
envs=(-e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1)
if [[ -n "${EXTRA_ENV:-}" ]]; then for e in $EXTRA_ENV; do envs+=(-e "$e"); done; fi
mkdir -p "$WORK/cache/$name"
mounts=(-v "$weights":/weights:ro -v "$WORK/cache/$name":/root/.cache -v "$RECIPE_DIR":/recipe:ro)
UAD=/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json
[[ -n "${UA_JSON:-}" ]] && mounts+=(-v "$(realpath "$UA_JSON"):$UAD:ro")
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then for m in $EXTRA_MOUNTS; do mounts+=(-v "$m"); done; fi
cmd=(vllm serve /weights --served-model-name "$MODEL" --host 0.0.0.0 --port "$port"
     --max-model-len 32768 --max-num-seqs "${MAXSEQS:-64}" --enable-prompt-tokens-details "$@")
# --userns=host is for daemons running with userns-remap (the host this was measured on); it is accepted
# and has no effect elsewhere.
docker run -d --name "$name" --userns=host --network host --ipc=host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video \
  "${envs[@]}" "${mounts[@]}" --entrypoint bash "$image" -c "${PRE:-true} && exec \"\$@\"" _ "${cmd[@]}"
