#!/usr/bin/env bash
# smoke.sh <port> [out.json]: the fixed 20-prompt smoke set at temperature 0, thinking off; with out.json it
# saves the answers so compare against the baseline with smoke20.py --diff.
source "$(dirname "$0")/common.sh"
python3 "$BENCH_DIR/smoke20.py" "http://127.0.0.1:$1" "${2:-}"
