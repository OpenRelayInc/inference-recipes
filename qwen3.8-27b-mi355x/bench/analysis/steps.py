# steps.py <trace.json.gz>: per-scheduler-step GPU time from vLLM's execute_context_X(T)_generation_Y(U) annotations
import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and "execute_context" in e.get("name", "") and e.get("cat") == (sys.argv[2] if len(sys.argv) > 2 else "gpu_user_annotation")]
cats = collections.defaultdict(list)
seen = set()
for e in ev:
    m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", e["name"])
    if not m: continue
    key = (e["ts"] // 1000, e["name"])
    if key in seen: continue
    seen.add(key)
    nc, tc, ng, tg = map(int, m.groups())
    kind = "decode-only" if nc == 0 else ("prefill-only" if ng == 0 else "mixed")
    cats[kind].append((e["dur"] / 1000, tc, ng, tg, e.get("cat")))
for k, v in cats.items():
    tot = sum(x[0] for x in v)
    print(f"{k:13} steps {len(v):5} total {tot:8.1f} ms  avg {tot/len(v):7.2f} ms  avg ctx tok {sum(x[1] for x in v)/len(v):7.0f}  avg gen reqs {sum(x[2] for x in v)/len(v):5.1f}  avg gen tok {sum(x[3] for x in v)/len(v):6.1f}  cats {set(x[4] for x in v)}")
dec = sorted(cats["decode-only"], key=lambda x: x[2])
if dec:
    import statistics
    by = collections.defaultdict(list)
    for x in dec: by[x[2] // 8 * 8].append(x[0])
    for b in sorted(by): print(f"  decode-only reqs {b:3}-{b+7:3}: n={len(by[b]):4} median {statistics.median(by[b]):6.2f} ms")
