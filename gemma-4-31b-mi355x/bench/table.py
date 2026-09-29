# table.py [results-dir]: one line per bench.sh / bench_text.sh result (rate-*.json one level down).
# Default results dir: $WORK/results, else ./work/results.
import json, glob, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("WORK", "work"), "results")
print(f"{'config':9} {'rps':>4} {'done':>5} {'fail':>4} {'ttft50':>7} {'ttft95':>7} {'tpot50':>6} {'tpot95':>6} {'e2e95':>7} {'in_tok/s':>9} {'out_tok/s':>9}")
for f in sorted(glob.glob(f"{root}/*/rate-*.json")):
    d = json.load(open(f)); c = f.split("/")[-2]; r = f.split("rate-")[1][:-5]
    print(f"{c:9} {r:>4} {d['completed']:>5} {d.get('failed', 0):>4} {d['median_ttft_ms']/1000:>7.2f} {d['p95_ttft_ms']/1000:>7.2f} "
          f"{d['median_tpot_ms']:>6.1f} {d['p95_tpot_ms']:>6.1f} {d['p95_e2el_ms']/1000:>7.1f} "
          f"{d['total_input_tokens']/d['duration']:>9.0f} {d['output_throughput']:>9.0f}")
