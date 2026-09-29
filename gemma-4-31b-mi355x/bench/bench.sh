#!/usr/bin/env bash
# bench.sh <name> <port> [rates...]: open-loop Poisson sweep against one engine, one `vllm bench serve` run
# per offered rate (requests/s). Shape of the production workload: input ~9.5K tokens (uniform 1.9K..17.1K),
# output ~380 (uniform 76..684), random tokens, no shared prefix.
#   DUR   seconds per rate (default 150; the final comparison used 300)
#   SEED  request-sampling seed (default 7; the final comparison ran 7 and 8)
# Results: $WORK/results/k-<name>-s<seed>/rate-<r>.{json,log}; table.py summarizes them.
# Random-token prompts give 97-98% MTP acceptance per position, far above real text. Kernel comparisons stay
# valid (every config sees the same prompts), but use bench_text.sh for anything that touches speculative decoding.
set -uo pipefail
source "$(dirname "$0")/common.sh"
name=$1 port=$2; shift 2
rates=("$@"); [[ ${#rates[@]} -eq 0 ]] && rates=(1.0 1.5 2.0 2.5 3.0)
seed=${SEED:-7}
out=$WORK/results/k-$name-s$seed; mkdir -p "$out"
for r in "${rates[@]}"; do
  n=$(python3 -c "print(int(float(\"$r\")*${DUR:-150}))")
  docker run --rm --userns=host --network host -v "$WEIGHTS":/tok:ro -v "$out":/out \
    --entrypoint vllm "$CLIENT_IMAGE" bench serve \
    --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer /tok --dataset-name random \
    --random-input-len 9500 --random-range-ratio 0.8 --random-output-len 380 \
    --num-prompts "$n" --request-rate "$r" --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir /out --result-filename "rate-$r.json" > "$out/rate-$r.log" 2>&1
  echo "$name seed $seed rate $r: $(grep -E "P95 TTFT|Median TPOT|Failed" "$out/rate-$r.log" | tr -s " " | tr "\n" " ")"
done
