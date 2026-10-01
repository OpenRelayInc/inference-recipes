# table.py <tag>...: one markdown row per results/<tag> directory (measure.sh output).
# prefill = second probe run x 8192 tok/s; c48/c16 = best lit_tpm of the two lockstep runs (both if > 5% apart);
# desync = staggered-start runs; knee = highest concurrency with TTFT p95 <= 2 s in each knee run.
import os, re, sys, glob
W = os.environ.get("WORK", "/mnt/nvme/ops/qwen38-opt") + "/results"
def rows(path):
    out = []
    for l in open(path):
        if re.match(r"^\d+ \| \d+ \|", l):
            f = [x.strip() for x in l.split("  statuses=")[0].split("|")]
            out.append(dict(conc=int(f[0]), out_tpm=float(f[4]), ttft95=float(f[7]), tpot=float(f[18]) if len(f) > 18 and f[18] else None,
                            lit=float(f[19]) if len(f) > 19 and f[19] else None))
    return out
def best(tag, name):
    rs = [r for p in sorted(glob.glob(f"{W}/{tag}/{name}-[12].log")) for r in rows(p)]
    return rs
def fmt(v, d=1): return "" if v is None else (f"{v/1000:.{d}f}K" if v > 1000 else f"{v:.{d}f}")
print("| stage | GSM8K | prefill tok/s | c48 TPM (lockstep) | c48 TTFT p95 | c48 TPOT ms | c16 TPM | c16 TPOT ms | c48 TPM desync | c48 TTFT p95 desync | knee (TPM) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for tag in sys.argv[1:]:
    g = open(f"{W}/{tag}/gsm8k.txt").read().split("=")[1].split("%")[0].strip() + "%" if os.path.exists(f"{W}/{tag}/gsm8k.txt") else ""
    pf = re.findall(r"\(req/s\): ([\d.]+)", open(f"{W}/{tag}/prefill.txt").read()) if os.path.exists(f"{W}/{tag}/prefill.txt") else []
    pre = f"{float(pf[-1])*8192/1000:.1f}K" if pf else ""
    c48 = best(tag, "c48"); c16 = best(tag, "c16")
    v = lambda rs, k: [r[k] if r[k] is not None else r["out_tpm"] for r in rs]
    def tp(rs):
        xs = v(rs, "lit")
        if not xs: return ""
        return fmt(max(xs)) if (max(xs) - min(xs)) / max(xs) <= 0.05 else " / ".join(fmt(x) for x in xs)
    d48 = rows(f"{W}/{tag}/c48d.log") if os.path.exists(f"{W}/{tag}/c48d.log") else []
    knees = []
    for p in sorted(glob.glob(f"{W}/{tag}/knee-[12].log")):
        rs = rows(p); ok = [r for r in rs if r["ttft95"] <= 2000]
        if rs: knees.append(f"{ok[-1]['conc']} ({fmt(ok[-1]['lit'] or ok[-1]['out_tpm'])})" if ok else "<2")
    print(f"| {tag} | {g} | {pre} | {tp(c48)} | {' / '.join(f'{r['ttft95']/1000:.2f} s' for r in c48)} | "
          f"{' / '.join(f'{r['tpot']:.1f}' for r in c48 if r['tpot'])} | {tp(c16)} | {' / '.join(f'{r['tpot']:.1f}' for r in c16 if r['tpot'])} | "
          f"{fmt(d48[0]['lit']) if d48 else ''} | {(str(round(d48[0]['ttft95']/1000,2))+' s') if d48 else ''} | {', '.join(knees)} |")
