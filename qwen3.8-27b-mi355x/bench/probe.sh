#!/usr/bin/env bash
# probe.sh <port> <concurrency> [input_len] [output_len] [num]: closed-loop vllm bench serve probe.
# Prefill metric: probe.sh <port> 16 8192 1 64, run twice; the second run counts.
source "$(dirname "$0")/common.sh"
port=$1 conc=$2 in=${3:-8192} outl=${4:-1} num=${5:-$(( conc * 4 ))}
docker run --rm --name "qopt-client-$port-$RANDOM" --userns=host --network host -v "$WEIGHTS":/tok:ro --entrypoint vllm "$CLIENT_IMAGE" bench serve \
  --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions --model "$MODEL" --tokenizer /tok \
  --dataset-name random --random-input-len "$in" --random-range-ratio 0 --random-output-len "$outl" \
  --num-prompts "$num" --max-concurrency "$conc" --seed $RANDOM --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95 2>&1 \
  | grep -E "Successful|Failed|Benchmark duration|Request throughput|Input token throughput|Mean TTFT|P95 TTFT" \
  | tr -s ' ' | sed "s/^/[$port c=$conc in=$in out=$outl] /"
