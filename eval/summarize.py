import json, sys
for path in sys.argv[1:]:
    print("==", path)
    for line in open(path):
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].startswith("{") and not line.startswith("{"):
            d = json.loads(parts[1])
            print(f"  {parts[0]:9s} err {d['top_error_m']:6.0f} m  truth rank {d['truth_rank']}/{d['n_candidates']}  top score {d['top_score']:.3f}")
