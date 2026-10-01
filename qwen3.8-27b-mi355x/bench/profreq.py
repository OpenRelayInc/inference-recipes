# profreq.py <port> <input_len> <n>: start_profile, send n concurrent random prompts of input_len tokens
# (1 output token each), stop_profile. The engine must run with --profiler-config (bench/README.md).
# Group the resulting trace with analysis/ktrace.py. Served model name from $MODEL (default qwen3.8-27b).
import os, sys, json, random, urllib.request, concurrent.futures as cf
port, L, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = os.environ.get("MODEL", "qwen3.8-27b")
base = f"http://127.0.0.1:{port}"
def post(path, body=None):
    req = urllib.request.Request(base + path, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900).read()
def one(i):
    rnd = random.Random(i * 1000 + L)
    toks = [rnd.randint(1000, 200000) for _ in range(L)]
    return post("/v1/completions", {"model": model, "prompt": toks, "max_tokens": 1, "temperature": 0})
post("/start_profile")
with cf.ThreadPoolExecutor(n) as ex: list(ex.map(one, range(n)))
post("/stop_profile")
print("done")
