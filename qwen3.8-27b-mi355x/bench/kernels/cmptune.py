# cmptune.py <pow2.csv> <exact.csv>: exact-M tuner picks against the power-of-two row aiter would pad up to.
import csv, sys
key = lambda r: (int(r["M"]), int(r["N"]), int(r["K"]))
a = {key(r): r for r in csv.DictReader(open(sys.argv[1]))}
b = {key(r): r for r in csv.DictReader(open(sys.argv[2]))}
for (m, n, k), r in sorted(b.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
    up = min(mm for mm in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if mm >= m)
    ra = a[(up, n, k)]
    print(f'{n:6} {k:6} M={m:4} exact {float(r["us"]):7.1f}us {r["libtype"]:6} | pow2 M={up:4} {float(ra["us"]):7.1f}us {ra["libtype"]:6} ratio {float(ra["us"])/float(r["us"]):.2f}')
