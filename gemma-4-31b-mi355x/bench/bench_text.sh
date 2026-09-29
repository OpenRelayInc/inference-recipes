#!/usr/bin/env bash
# bench_text.sh <name> <port> [rates...]: bench.sh with natural-text prompts ($WORK/data/text_bench.jsonl from
# mkdata.py, input uniform ~1.9K..17.1K tokens) and 380 forced output tokens, so MTP acceptance is realistic.
# DUR seconds per rate (default 150). Results: $WORK/results/text-<name>/rate-<r>.{json,log}.
set -uo pipefail
source "$(dirname "$0")/common.sh"
name=$1 port=$2; shift 2
rates=("$@"); [[ ${#rates[@]} -eq 0 ]] && rates=(2.5 3.0 3.5)
data=${DATA:-$WORK/data/text_bench.jsonl}
[[ -f "$data" ]] || { echo "missing $data: run mkdata.py first (bench/README.md)" >&2; exit 1; }
out=$WORK/results/text-$name; mkdir -p "$out"
for r in "${rates[@]}"; do
  n=$(python3 -c "print(int(float(\"$r\")*${DUR:-150}))")
  docker run --rm --userns=host --network host -v "$WEIGHTS":/tok:ro -v "$out":/out -v "$(dirname "$data")":/data:ro \
    --entrypoint vllm "$CLIENT_IMAGE" bench serve \
    --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer /tok --dataset-name custom --dataset-path "/data/$(basename "$data")" \
    --custom-output-len 380 --ignore-eos --skip-chat-template \
    --num-prompts "$n" --request-rate "$r" --seed 7 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir /out --result-filename "rate-$r.json" > "$out/rate-$r.log" 2>&1
  echo "$name rate $r done: $(grep -E "P95 TTFT|Median TPOT|Total input tokens|Failed" "$out/rate-$r.log" | tr -s " " | tr "\n" " ")"
done
