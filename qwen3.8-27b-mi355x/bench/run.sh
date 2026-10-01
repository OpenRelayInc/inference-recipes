#!/usr/bin/env bash
# run.sh <tag> <port> [full]: gate (smoke + GSM8K) then measure.sh, logs to $WORK/results/measure-<tag>.out.
source "$(dirname "$0")/common.sh"
"$BENCH_DIR/gate.sh" "$1" "$2" > "$WORK/results/measure-$1.out" 2>&1
"$BENCH_DIR/measure.sh" "$1" "$2" ${3:-} >> "$WORK/results/measure-$1.out" 2>&1
