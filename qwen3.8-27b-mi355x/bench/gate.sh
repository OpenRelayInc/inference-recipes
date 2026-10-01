#!/usr/bin/env bash
# gate.sh <tag> <port>: smoke20 (saved, diffed against the baseline-B answers if present) and GSM8K first 250 at c=32.
source "$(dirname "$0")/common.sh"
tag=$1 port=$2; out=$WORK/results/$tag; mkdir -p "$out" "$WORK/results/smoke"
echo "[$tag] smoke: $("$BENCH_DIR/smoke.sh" "$port" "$WORK/results/smoke/$tag.json")"
[[ -f $WORK/results/smoke/B.json && $tag != B ]] && echo "[$tag] vs B: $(python3 "$BENCH_DIR/smoke20.py" --diff "$WORK/results/smoke/B.json" "$WORK/results/smoke/$tag.json")"
echo "[$tag] $(python3 "$BENCH_DIR/gsm8k.py" "http://127.0.0.1:$port" 250 32)" | tee "$out/gsm8k.txt"
