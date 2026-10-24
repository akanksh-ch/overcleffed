"""Run Clef-flash (or mock/greedy) on overcooked_ai. See README."""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from overcleffed.env import History, build_state_payload, layout_summary, make_env, serialize_state
from overcleffed.policy import ClefPolicy, GreedyPolicy, MockPolicy, RawActions


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_episode(ep, args, policy, log, pool):
    env = make_env(args.layout, args.horizon)
    policy.reset(env)
    space = policy.space
    layout = layout_summary(env.mdp)
    hist = History(args.history_k)
    state, score, soups, lats, late = env.state, 0, 0, [], 0
    ser = serialize_state(state)
    hist.push(ser)
    done, tick = False, args.tick_hz and 1.0 / args.tick_hz
    while not done:
        t0 = time.perf_counter()
        payload = build_state_payload(layout, ser, hist, score, args.horizon)
        payload["layout"] = layout
        fut = pool.submit(policy.act, payload, state) if tick else None
        timed_out, err = False, None
        try:
            if tick:
                names, probs, _ = fut.result(timeout=tick)
            else:
                names, probs, _ = policy.act(payload, state)
        except Exception as e:  # late or failed -> stay
            timed_out, err = True, f"{type(e).__name__}: {e}"
            names, probs = [space.stay] * 2, None
            late += 1
        lat = time.perf_counter() - t0
        lats.append(lat)
        joint = tuple(space.to_env_action(n, state, i) for i, n in enumerate(names))
        state, r, done, info = env.step(joint)
        score += r
        soups += int(round(sum(info.get("sparse_reward_by_agent", [r])) / 20))
        log.write(json.dumps({"ep": ep, "t": ser["t"], "actions": names, "probs": probs,
                              "reward": r, "score": score, "latency_s": round(lat, 4),
                              "late": timed_out, "err": err}) + "\n")
        ser = serialize_state(state)
        # attach actions/reward to previous entry, then push new state
        hist.actions[-1], hist.rewards[-1] = tuple(names), r
        hist.push(ser)
        if tick:
            time.sleep(max(0, tick - (time.perf_counter() - t0)))
    return {"ep": ep, "soups": soups, "reward": score, "steps": ser["t"], "late_or_failed": late,
            "mean_latency_s": statistics.mean(lats), "p95_latency_s": pct(lats, 0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--history-k", type=int, default=5)
    ap.add_argument("--decode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tick-hz", type=float, default=0, help="0 = lockstep; >0 = real-time, late -> stay")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--backend", choices=["clef", "mock", "greedy"], default="clef")
    ap.add_argument("--mock", action="store_true", help="alias for --backend mock")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mock:
        args.backend = "mock"
    policy = {"clef": lambda: ClefPolicy(args.url, args.decode, args.temperature),
              "mock": lambda: MockPolicy(args.seed), "greedy": GreedyPolicy}[args.backend]()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summaries = []
    with open(args.out + ".jsonl", "w") as log, ThreadPoolExecutor(1) as pool:
        for ep in range(args.episodes):
            s = run_episode(ep, args, policy, log, pool)
            summaries.append(s)
            print(json.dumps(s), file=sys.stderr)
    tot = {"args": vars(args), "episodes": summaries,
           "soups_total": sum(s["soups"] for s in summaries),
           "reward_total": sum(s["reward"] for s in summaries),
           "mean_latency_s": statistics.mean(s["mean_latency_s"] for s in summaries),
           "p95_latency_s": max(s["p95_latency_s"] for s in summaries)}
    json.dump(tot, open(args.out + ".summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in tot.items() if k not in ("args", "episodes")}))


if __name__ == "__main__":
    main()
