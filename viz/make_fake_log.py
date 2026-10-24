"""Make a fake log with synthetic probs/latency from a real action log. Usage: make_fake_log.py runs/greedy.jsonl out.jsonl"""
import json, random, sys
N = ["up", "down", "left", "right", "stay", "interact"]
rng = random.Random(0)
out = open(sys.argv[2], "w")
for line in open(sys.argv[1]):
    r = json.loads(line)
    probs = []
    for a in r["actions"]:
        w = [rng.random() * 0.15 for _ in N]
        w[N.index(a)] += rng.uniform(0.2, 1.5)
        s = sum(w); probs.append({n: x / s for n, x in zip(N, w)})
    r["probs"] = probs
    r["latency_s"] = round(rng.lognormvariate(-2.5, 0.6), 4)
    r["late"] = r["latency_s"] > 0.2
    out.write(json.dumps(r) + "\n")
