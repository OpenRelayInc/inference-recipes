#!/usr/bin/env bash
# waitready.sh <name> <port> [timeout_s]: waits for /health 200, prints seconds since launch.sh started it.
source "$(dirname "$0")/common.sh"
name=$1 port=$2 to=${3:-1500}
t0=$(cat "$WORK/cache/.launched-$name")
while :; do
  if curl -sf -o /dev/null "127.0.0.1:$port/health"; then echo "$name ready in $(( $(date +%s) - t0 )) s"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "$name exited"; docker logs --tail 40 "$name" 2>&1; exit 1; fi
  (( $(date +%s) - t0 > to )) && { echo "$name not ready after $to s"; exit 1; }
  sleep 3
done
