#!/usr/bin/env bash
# tpm.sh <port> <out.log> <steps> [seconds] [warmup]: closed-loop 5:1 ladder (tpm_bench.py) against a local
# engine (STAGGER=<s> spreads stream starts over s seconds): ~2,500 prompt tokens of unique random text, exactly 500 output tokens (ignore_eos), thinking off.
source "$(dirname "$0")/common.sh"
port=$1 out=$2 steps=$3 secs=${4:-140} warm=${5:-20}
mkdir -p "$(dirname "$out")"
docker run --rm --name "qopt-client-$port-$RANDOM" --userns=host --network host -e OR_KEY=x \
  -e TPM_URL="http://127.0.0.1:$port/v1/chat/completions" -e TPM_MODEL="$MODEL" \
  -v "$BENCH_DIR":/bench:ro --entrypoint python3 "$CLIENT_IMAGE" /bench/tpm_bench.py \
  --steps "$steps" --seconds "$secs" --warmup "$warm" --words "$WORDS" --pause 5 --stagger "${STAGGER:-0}" > "$out" 2>&1
grep -E "^[0-9]+ [|] [0-9]+ [|]" "$out" | sed "s/  statuses=.*//" | awk -F" [|] " -v p=$port '{printf "[%s] conc=%s out_tpm=%s lit_tpm=%s ttft_p50=%s ttft_p95=%s tpot_mean=%s decode_tps=%s n429/err/trunc=%s/%s/%s\n",p,$1,$5,$20,$7,$8,$19,$12,$15,$16,$17}'
