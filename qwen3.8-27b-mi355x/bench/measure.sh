#!/usr/bin/env bash
# measure.sh <tag> <port> [full]: the per-experiment metric set on one engine. Results in $WORK/results/<tag>/.
#   saturated: c=48 twice (primary metric: output TPM; also mean TPOT at 48)
#   decode:    c=16 twice (mean TPOT at 16)
#   prefill:   16 concurrent 8K prompts, 1 output token, twice (second counts)
#   desync:    c=48 and c=16 once each with staggered starts (STAGGER 12 s / 4 s), files c48d.log c16d.log
#   full:      also the knee ladder 2,4,8,16,24,32,48,64 (TTFT p95 <= 2 s), twice
source "$(dirname "$0")/common.sh"
tag=$1 port=$2 full=${3:-}
out=$WORK/results/$tag; mkdir -p "$out"
for i in 1 2; do "$BENCH_DIR/probe.sh" "$port" 16 8192 1 64 | tee -a "$out/prefill.txt"; done
for i in 1 2; do "$BENCH_DIR/tpm.sh" "$port" "$out/c48-$i.log" 48 | tee -a "$out/summary.txt"; done
for i in 1 2; do "$BENCH_DIR/tpm.sh" "$port" "$out/c16-$i.log" 16 | tee -a "$out/summary.txt"; done
STAGGER=12 "$BENCH_DIR/tpm.sh" "$port" "$out/c48d.log" 48 150 30 | sed "s/^/desync /" | tee -a "$out/summary.txt"
STAGGER=4 "$BENCH_DIR/tpm.sh" "$port" "$out/c16d.log" 16 150 30 | sed "s/^/desync /" | tee -a "$out/summary.txt"
if [[ $full == full ]]; then
  for i in 1 2; do "$BENCH_DIR/tpm.sh" "$port" "$out/knee-$i.log" 2,4,8,16,24,32,48,64 | tee -a "$out/knee.txt"; done
fi
