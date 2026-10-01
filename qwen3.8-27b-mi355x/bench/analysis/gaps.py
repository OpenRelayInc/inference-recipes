# gaps.py <trace> [mixed|decode]: idle gaps between kernels inside step windows, grouped by the kernel pair around them
import gzip, json, sys, re, collections, bisect
d = json.load(gzip.open(sys.argv[1])); want = sys.argv[2] if len(sys.argv) > 2 else "mixed"
ev = d["traceEvents"]; win = []
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation":
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", e.get("name", ""))
        if m and ((int(m.group(1)) == 0) == (want == "decode")): win.append((e["ts"], e["ts"] + e["dur"]))
win.sort()
ks = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
starts = [k["ts"] for k in ks]
g = collections.Counter(); gc = collections.Counter(); tot = 0; span = 0
for a, b in win:
    i = bisect.bisect_left(starts, a); prev_end = a; prev = "<start>"
    while i < len(ks) and ks[i]["ts"] < b:
        gap = ks[i]["ts"] - prev_end
        if gap > 50:
            key = (prev[:50], ks[i]["name"][:50]); g[key] += gap; gc[key] += 1; tot += gap
        prev_end = max(prev_end, ks[i]["ts"] + ks[i]["dur"]); prev = ks[i]["name"]; i += 1
    span += b - a
print(f"{want}: {len(win)} steps, gaps>50us total {tot/1000:.1f} ms of {span/1000:.1f} ms ({tot/span*100:.1f}%)")
for k, v in g.most_common(12): print(f"  {v/1000:7.1f} ms x{gc[k]:5}  {k[0]}  ->  {k[1]}")
