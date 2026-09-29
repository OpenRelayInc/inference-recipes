#!/usr/bin/env bash
# smoke.sh <port>: one chat prompt at temperature 0, prints the answer. A healthy engine names Canberra and 391.
source "$(dirname "$0")/common.sh"
curl -s "localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the capital of Australia, and what is 17*23? Answer in one short sentence."}],"max_tokens":60,"temperature":0}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
