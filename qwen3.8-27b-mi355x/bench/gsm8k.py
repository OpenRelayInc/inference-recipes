# gsm8k.py <base-url> [n=250] [conc=32] [offset=0]: the correctness gate. First n GSM8K test problems via chat,
# temperature 0, thinking off (chat_template_kwargs enable_thinking=false), scored on '#### <n>' or the last
# number. A config fails the gate below 96.5% on 250 problems. The test split is cached at $GSM8K (default
# $WORK/data/gsm8k_test.jsonl). Served model name from $MODEL (default qwen3.8-27b).
import json, re, sys, os, asyncio, time, urllib.request
base = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 250
conc = int(sys.argv[3]) if len(sys.argv) > 3 else 32
off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
path = os.environ.get("GSM8K") or os.path.join(os.environ.get("WORK", "work"), "data", "gsm8k_test.jsonl")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    urllib.request.urlretrieve(URL, path)
rows = [json.loads(l) for l in open(path)][off:off + n]
hdr = {"Content-Type": "application/json"}

def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None

def ask(q):
    body = {"model": os.environ.get("MODEL", "qwen3.8-27b"),
            "messages": [{"role": "user", "content": q + "\nSolve step by step, then give the final answer as a plain number on the last line in the form: #### <number>"}],
            "max_tokens": 2048, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), hdr)
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"].get("content") or ""

async def main():
    sem = asyncio.Semaphore(conc); ok = 0; err = 0
    async def one(r):
        nonlocal ok, err
        async with sem:
            try: out = await asyncio.to_thread(ask, r["question"])
            except Exception: err += 1; return
        m = re.search(r"####\s*([-\d.,$]+)", out); pred = num(m.group(1)) if m else num(out)
        if pred is not None and abs(pred - num(r["answer"].split("####")[-1])) < 1e-6: ok += 1
    t = time.time(); await asyncio.gather(*[one(r) for r in rows])
    print(f"{base} GSM8K {ok}/{len(rows)} = {ok/len(rows)*100:.1f}%  err={err}  {time.time()-t:.0f}s")
asyncio.run(main())
