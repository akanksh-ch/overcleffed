"""Run Clef-flash (or mock/greedy) on overcooked_ai. See README."""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from overcleffed.env import History, build_state_payload, layout_summary, make_env, serialize_state
from overcleffed.policy import (ClefPolicy, GreedyPolicy, MacroPolicy, MockPolicy, RawActions,
                                clef_chooser, mock_chooser)


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_episode(ep, args, policy, log, pool):
    env = make_env(args.layout, args.horizon)
    policy.reset(env)
    space = policy.space
    macro = args.actions == "macro"
    layout = layout_summary(env.mdp)
    hist = History(max(args.history_k, 1))
    state, score, soups, lats, late = env.state, 0, 0, [], 0
    calls, toks, confs, call_lats = 0, [], [], []
    ag, cag, cdis = [], [], []
    ser = serialize_state(state)
    hist.push(ser)
    done, tick = False, args.tick_hz and 1.0 / args.tick_hz
    while not done:
        t0 = time.perf_counter()
        payload = {"score": score} if macro else build_state_payload(layout, ser, hist, score, args.horizon)
        fut = pool.submit(policy.act, payload, state) if tick else None
        timed_out, err, info = False, None, {}
        try:
            if tick:
                names, probs, info = fut.result(timeout=tick)
            else:
                names, probs, info = policy.act(payload, state)
        except Exception as e:  # late or failed -> stay
            timed_out, err = True, f"{type(e).__name__}: {e}"
            names, probs, info = [space.stay] * 2, None, {}
            late += 1
        lat = time.perf_counter() - t0
        lats.append(lat)
        joint = tuple(space.to_env_action(n, state, i) for i, n in enumerate(names))
        state, r, done, einfo = env.step(joint)
        score += r
        soups += int(round(sum(einfo.get("sparse_reward_by_agent", [r])) / 20))
        log.write(json.dumps(default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), obj={"ep": ep, "t": ser["t"], "actions": names, "probs": probs,
                              "reward": r, "score": score, "latency_s": round(lat, 4),
                              "late": timed_out, "err": err, **info}) + "\n")
        log.flush()
        if info.get("clef_called"):
            for a_, c_ in zip(info.get("agree") or [], info.get("conf_by_chef") or []):
                if a_ is not None:
                    ag.append(a_)
                    if c_ is not None:
                        (cag if a_ else cdis).append(c_)
            calls += 1
            call_lats.append(lat)
            if info.get("input_tokens"):
                toks.append(info["input_tokens"])
            if info.get("confidence") is not None:
                confs.append(info["confidence"])
            print(f"t={ser['t']} call lat={lat:.1f}s tok={info.get('input_tokens')} "
                  f"sub={info.get('subgoal')} conf={info.get('confidence')} score={score}", file=sys.stderr, flush=True)
        ser = serialize_state(state)
        # attach actions/reward to previous entry, then push new state
        hist.actions[-1], hist.rewards[-1] = tuple(names), r
        hist.push(ser)
        if tick or args.sim_hz:
            period = tick or 1.0 / args.sim_hz
            time.sleep(max(0, period - (time.perf_counter() - t0)))
    return {"ep": ep, "soups": soups, "reward": score, "steps": ser["t"], "late_or_failed": late,
            "mean_latency_s": statistics.mean(lats), "p95_latency_s": pct(lats, 0.95),
            "n_clef_calls": calls, "mean_input_tokens": statistics.mean(toks) if toks else None,
            "mean_confidence": statistics.mean(confs) if confs else None,
            "agreement_rate": statistics.mean(ag) if ag else None, "n_agree_compared": len(ag),
            "conf_when_agree": statistics.mean(cag) if cag else None,
            "conf_when_disagree": statistics.mean(cdis) if cdis else None,
            "mean_call_latency_s": statistics.mean(call_lats) if call_lats else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--history-k", type=int, default=None, help="raw: 5 states; macro: 2 subgoals")
    ap.add_argument("--question-mode", choices=["choice", "noul"], default="choice")
    ap.add_argument("--roles", action="store_true", help="state chef roles in prompt (macro)")
    ap.add_argument("--actions", choices=["raw", "macro"], default="raw")
    ap.add_argument("--decode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tick-hz", type=float, default=0, help="0 = lockstep; >0 = real-time, late -> stay")
    ap.add_argument("--async-clef", action="store_true", help="experimental independent per-chef macro requests")
    ap.add_argument("--sim-hz", type=float, default=0, help="pace simulation without timing out inference")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--backend", choices=["clef", "mock", "greedy"], default="clef")
    ap.add_argument("--mock", action="store_true", help="alias for --backend mock")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.async_clef and (args.actions != "macro" or args.backend != "clef" or args.mock or args.tick_hz or args.sim_hz <= 0):
        ap.error("--async-clef requires --actions macro --backend clef --sim-hz > 0 and no --tick-hz")
    if args.mock:
        args.backend = "mock"
    if args.history_k is None:
        args.history_k = 2 if args.actions == "macro" else 5
    if args.actions == "macro" and args.backend != "greedy":
        policy = MacroPolicy(clef_chooser(args.url, args.decode, args.temperature, question_mode=args.question_mode) if args.backend == "clef"
                             else mock_chooser(args.seed), args.history_k, args.roles)
    else:
      policy = {"clef": lambda: ClefPolicy(args.url, args.decode, args.temperature),
              "mock": lambda: MockPolicy(args.seed), "greedy": GreedyPolicy}[args.backend]()
    if args.async_clef:
        from overcleffed.async_policy import AsyncMacroPolicy
        policy = AsyncMacroPolicy([clef_chooser(args.url, args.decode, args.temperature, timeout=10,
                                                question_mode=args.question_mode) for _ in range(2)],
                                  args.history_k, args.roles)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summaries = []
    with open(args.out + ".jsonl", "w", buffering=1) as log, ThreadPoolExecutor(1) as pool:
        for ep in range(args.episodes):
            s = run_episode(ep, args, policy, log, pool)
            summaries.append(s)
            print(json.dumps(s), file=sys.stderr)
    if args.async_clef:
        policy.close()
    tot = {"args": vars(args), "episodes": summaries,
           "soups_total": sum(s["soups"] for s in summaries),
           "reward_total": sum(s["reward"] for s in summaries),
           "mean_latency_s": statistics.mean(s["mean_latency_s"] for s in summaries),
           "p95_latency_s": max(s["p95_latency_s"] for s in summaries)}
    json.dump(tot, open(args.out + ".summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in tot.items() if k not in ("args", "episodes")}))


if __name__ == "__main__":
    main()
