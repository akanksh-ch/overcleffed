"""Run Clef chef + Clef assistant on Collab-Overcooked tasks. Run from the benchmark's src/.

python run_clef.py --orders boiled_egg baked_bell_pepper_soup --horizon 120 --out runs_clef
Writes <out>/<order>.json: per-step map/holdings/utensils/reward + per-decision Clef probs.
Success = order delivered within horizon (the paper's fix_task mode).
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner

from clef_agent import ClefAgents
from planner import Planner
from gemma_planner import GemmaPlanner


def make(mdp, actor):
    cl = mdp.get_counter_locations()
    params = {"start_orientations": False, "wait_allowed": True, "counter_goals": cl,
              "counter_drop": cl, "counter_pickup": cl, "same_motion_goals": True}
    mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, params, force_compute=True).ml_action_manager
    a = ClefAgents(mlam, "new_env", model="clef", actor=actor)
    a.set_mdp(mdp)
    return a


def run(order, horizon, hier=False, planner="script"):
    mdp = OvercookedGridworld.from_layout_name("new_env")
    mdp.start_order_list, mdp.one_task_mode = [order], True
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    decisions = []
    team = AgentGroup(make(mdp, "chef"), make(mdp, "assistant"))
    plan = (GemmaPlanner if planner == "gemma" else Planner)(mdp, order) if hier else None
    for a in team.agents:
        a.log, a.planner_ = decisions, plan
    team.reset()
    env.reset()
    steps, total, t0 = [], 0, time.time()
    for t in range(horizon):
        s = env.state
        a_t, parm = team.joint_action(s)
        team.reset_dialogue()
        _, r, done, _ = env.step(a_t, parm)
        total += r
        steps.append({"t": t, "map": mdp.state_string(s).replace("ø", "o"),
                      "held": [p.held_object.name if p.held_object else None for p in s.players],
                      "utensils": {k: v for k, v in mdp.get_utensil_states(s).items() if v},
                      "pos": [list(p.position) for p in s.players],
                      "objects": [[o.name, list(o.position),
                                   (o.state[0] if isinstance(o.state[0], (str, list)) else None) if o.state else None,
                                   (o.state[2] if o.state and len(o.state) > 2 else None)]
                                  for o in s.objects.values()],
                      "skill": [a.current_ml_action for a in team.agents],
                      "goal": [g[i] for g in [plan.goals(s)] for i in (0, 1)] if plan else None, "reward": r})
        if r > 0 or done:
            break
    return {"order": order, "success": total > 0, "steps": len(steps), "reward": total,
            "wall_s": time.time() - t0, "decisions": decisions, "trajectory": steps,
            "planner_calls": getattr(plan, "calls", [])}


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--orders", nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out", default="runs_clef")
    ap.add_argument("--planner", choices=["script", "gemma"], default="script")
    ap.add_argument("--hier", action="store_true", help="planner sets goals, Clef picks skills, wait masked")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for o in args.orders:
        res = run(o, args.horizon, args.hier, args.planner)
        json.dump(res, open(f"{args.out}/{o}.json", "w"))
        lat = [d["latency_s"] for d in res["decisions"]]
        print(json.dumps({"order": o, "success": res["success"], "steps": res["steps"],
                          "clef_calls": len(lat), "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
                          "errors": sum(1 for d in res["decisions"] if d["err"]),
                          "planner_calls": len(res["planner_calls"]),
                          "planner_latency_s": round(sum(c["latency_s"] for c in res["planner_calls"]) / max(len(res["planner_calls"]), 1), 2),
                          "planner_errors": sum(1 for c in res["planner_calls"] if c["err"])}), flush=True)
