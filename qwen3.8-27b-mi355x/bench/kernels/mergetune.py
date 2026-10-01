# mergetune.py <pow2.csv> <exact.csv> <out.csv> [min_ratio=1.05]: the power-of-two rows plus every exact-M row that
# beats the power-of-two row aiter would otherwise pad up to by at least min_ratio.
import csv, sys
key = lambda r: (int(r["M"]), int(r["N"]), int(r["K"]))
rows = list(csv.DictReader(open(sys.argv[1])))
a = {key(r): r for r in rows}
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 1.05
kept = 0
for r in csv.DictReader(open(sys.argv[2])):
    m, n, k = key(r)
    up = min(mm for mm in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if mm >= m)
    if float(a[(up, n, k)]["us"]) / float(r["us"]) >= thr:
        rows.append(r); kept += 1
rows.sort(key=lambda r: (int(r["N"]), int(r["K"]), int(r["M"])))
w = csv.DictWriter(open(sys.argv[3], "w", newline=""), fieldnames=list(rows[0].keys()), lineterminator="\n")
w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows ({kept} exact-M rows kept at ratio >= {thr})")
