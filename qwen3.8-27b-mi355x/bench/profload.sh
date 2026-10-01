#!/usr/bin/env bash
# profload.sh <port> <conc> [warm_s]: closed-loop 5:1 load (tpm_bench.py) at <conc>; after warm_s seconds
# (default 60) POST /start_profile, wait 10 s, POST /stop_profile. The engine must run with --profiler-config;
# its active_iterations bounds the capture. Traces land in the engine's /root/.cache/prof.
source "$(dirname "$0")/common.sh"
port=$1 conc=$2 warm=${3:-60}
"$BENCH_DIR/tpm.sh" "$port" "$WORK/results/profload-$port-c$conc.log" "$conc" $((warm + 40)) 5 &
sleep "$warm"
curl -s -X POST "127.0.0.1:$port/start_profile"; sleep 10; curl -s -X POST "127.0.0.1:$port/stop_profile"
wait
