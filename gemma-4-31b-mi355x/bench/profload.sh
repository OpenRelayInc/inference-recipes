#!/usr/bin/env bash
# profload.sh <port> <rate> [warm_s]: open-loop load at the workload's shape; after warm_s seconds (default 60)
# POST /start_profile, wait 20 s, POST /stop_profile. The engine must run with --profiler-config (bench/README.md);
# its active_iterations bounds the capture. Traces land in the engine's /root/.cache/prof, which launch.sh
# maps to $WORK/cache/<engine-name>/prof. Split them per scheduler step with analysis/steps.py and stepkern.py.
source "$(dirname "$0")/common.sh"
port=$1 rate=$2 warm=${3:-60}
log=$WORK/results/profload-$port.log; mkdir -p "$(dirname "$log")"
n=$(python3 -c "print(int($rate*($warm+60)))")
docker run --rm --userns=host --network host -v "$WEIGHTS":/tok:ro --entrypoint vllm "$CLIENT_IMAGE" bench serve \
  --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions --model "$MODEL" --tokenizer /tok \
  --dataset-name random --random-input-len 9500 --random-range-ratio 0.8 --random-output-len 380 \
  --num-prompts "$n" --request-rate "$rate" --seed 11 > "$log" 2>&1 &
sleep "$warm"
curl -s -X POST "localhost:$port/start_profile"; sleep 20; curl -s -X POST "localhost:$port/stop_profile"
wait
grep -E "P95 TTFT|Median TPOT" "$log"
