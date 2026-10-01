# stepkern.py <trace.json.gz> [decode|mixed] [topn]: kernel time inside decode-only (or mixed) step windows
import gzip, json, sys, re, collections, bisect
d = json.load(gzip.open(sys.argv[1])); want = sys.argv[2] if len(sys.argv) > 2 else "decode"
ev = d["traceEvents"]
win = []
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation":
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", e.get("name", ""))
        if m and ((int(m.group(1)) == 0) == (want == "decode")): win.append((e["ts"], e["ts"] + e["dur"]))
win.sort(); starts = [w[0] for w in win]
k = collections.Counter(); c = collections.Counter(); tot = 0
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i >= 0 and e["ts"] < win[i][1]:
            k[e["name"][:100]] += e["dur"]; c[e["name"][:100]] += 1; tot += e["dur"]
span = sum(b - a for a, b in win)
print(f"{want}: {len(win)} steps, window {span/1000:.1f} ms, kernels {tot/1000:.1f} ms ({tot/span*100:.0f}% busy), per step {tot/len(win)/1000:.2f} ms kernels")
for n, v in k.most_common(int(sys.argv[3]) if len(sys.argv) > 3 else 20):
    print(f"  {v/len(win)/1000:7.3f} ms/step x{c[n]/len(win):5.1f}  {n}")
