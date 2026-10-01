import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in sorted(rows, key=lambda r: (int(r["N"]), int(r["K"]), int(r["M"]))):
    print(f'{r["N"]:>6} {r["K"]:>6} {r["M"]:>6} {r["libtype"]:7} {float(r["us"]):10.1f}us {float(r["tflops"]):8.1f}TF err={r["errRatio"]} {r["kernelName"][:58]}')
