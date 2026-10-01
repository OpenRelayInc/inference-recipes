# kparent.py <trace> <kernel-substring>: which CPU ops (and input shapes) launch a kernel
import gzip, json, sys, collections, bisect
d = json.load(gzip.open(sys.argv[1])); ev = d["traceEvents"]; pat = sys.argv[2]
corr = {}
for e in ev:
    if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e and "correlation" in e["args"]:
        corr[e["args"]["correlation"]] = e
ops = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
res = collections.Counter()
for e in ev:
    if e.get("cat") == "kernel" and pat in e.get("name", ""):
        r = corr.get(e["args"].get("correlation"))
        if not r: res["?"] += e["dur"]; continue
        # innermost cpu_op enclosing the runtime call
        best = None
        for o in ops:
            if o["ts"] <= r["ts"] and o["ts"] + o["dur"] >= r["ts"] + r["dur"]:
                if best is None or o["dur"] < best["dur"]: best = o
            if o["ts"] > r["ts"]: break
        key = (best["name"], str(best.get("args", {}).get("Input Dims", ""))[:120]) if best else ("none", "")
        res[key] += e["dur"]
for k, v in res.most_common(8): print(f"{v/1000:8.1f} ms  {k}")
