#!/usr/bin/env bash
# launch.sh <name> <gpu> <port> <image> [engine args...]
# One TP1 Qwen3.8-27B engine on one MI355X (rocm-smi index 4..7), detached, in container <name> (must start
# with qopt-). The container gets pod-equivalent limits and exactly the render and card node of <gpu>, so the
# engine sees one device and needs no *_VISIBLE_DEVICES. The port binds to 127.0.0.1 only.
#   EXTRA_MOUNTS="h:c .."  extra bind mounts (tuned tables)
#   EXTRA_ENV="K=V .."     extra environment
#   PRE="cmd"              shell command before vllm serve; this recipe is mounted at /recipe
#   CACHE=<dir>            host dir for /root/.cache (default $WORK/cache/<name>)
# Base args always passed: the production flags that are not under test (see engine_args.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
name=$1 gpu=$2 port=$3 image=$4; shift 4
[[ $name == qopt-* ]] || { echo "container name must start with qopt-" >&2; exit 1; }
devs=$(gpu_devs "$gpu")
docker rm -f "$name" >/dev/null 2>&1 || true
if curl -s -o /dev/null "127.0.0.1:$port/v1/models"; then echo "port $port busy" >&2; exit 1; fi
for c in $(docker ps --format "{{.Names}}" | grep '^qopt-' || true); do
  if docker inspect "$c" --format "{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}" | grep -qx "${devs%% *}"; then
    echo "GPU $gpu held by $c" >&2; exit 1
  fi
done
devargs=(--device /dev/kfd); for d in $devs; do devargs+=(--device "$d"); done
envs=(-e VLLM_ROCM_USE_AITER=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/root/.cache/triton)
if [[ -n "${EXTRA_ENV:-}" ]]; then for e in $EXTRA_ENV; do envs+=(-e "$e"); done; fi
cache=${CACHE:-$WORK/cache/$name}; mkdir -p "$cache"
# The daemon runs with userns-remap: container root is host uid 100000 (/etc/subuid, dockremap).
sudo chown -R 100000:100000 "$cache"
mounts=(-v "$WEIGHTS":/weights:ro -v "$cache":/root/.cache -v "$RECIPE_DIR":/recipe:ro)
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then for m in $EXTRA_MOUNTS; do mounts+=(-v "$m"); done; fi
cmd=(vllm serve /weights --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 "$@")
docker run -d --name "$name" -p "127.0.0.1:$port:8000" \
  --memory 300g --memory-swap 300g --cpus 32 --shm-size 8g --security-opt no-new-privileges \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --group-add 44 --group-add 992 \
  "${devargs[@]}" "${envs[@]}" "${mounts[@]}" --entrypoint bash "$image" -c "${PRE:-true} && exec \"\$@\"" _ "${cmd[@]}"
date +%s > "$WORK/cache/.launched-$name"
