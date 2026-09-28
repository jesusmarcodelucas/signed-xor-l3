import sys, json, ae
v, s = sys.argv[1], int(sys.argv[2])
r = ae.run(v, seed=s)
json.dump(r, open(f"res_{v}_{s}.json", "w"))
