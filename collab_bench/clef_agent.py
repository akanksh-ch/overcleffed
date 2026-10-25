"""Clef as the skill picker inside Collab-Overcooked (EMNLP 2025).

Subclasses the benchmark's LLMAgents and replaces only `generate_ml_action`:
enumerate grounded skills, keep the ones the benchmark's own validator accepts,
ask Clef one `choice` question, return the chosen skill string. The benchmark
still executes skills with its own path planner, so execution matches the paper.

Deviations from the paper's protocol (logged in the decision thread):
- No natural-language chat/requests between agents. Both see the recipe instead.
- Invalid skills are masked out up front instead of fixed by self-correction.
"""
import json
import os
import re
import time

import requests

from collab.collab import LLMAgents

URL = os.environ.get("CLEF_URL", "http://127.0.0.1:8080")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
OPS = {"chef": ["cook", "bake"], "assistant": ["cut", "stir"]}
ROLE = {"chef": "Chef: works pot/oven, fills dishes, serves. Cannot reach dispensers.",
        "assistant": "Assistant: works chopping board/blender, fetches ingredients and dishes, "
                     "passes items to the chef via the shared counter. Cannot serve."}


def recipe_text(order):
    for f in os.listdir(os.path.join(PROMPT_DIR, "recipe")):
        if order in f.lower():
            return open(os.path.join(PROMPT_DIR, "recipe", f)).read().strip()
    return ""


def tree_items(mdp, order):
    """All intermediate/raw items in the order's recipe tree, plus the order itself."""
    out, todo = {order}, [order]
    while todo:
        x = todo.pop()
        for utensil in mdp.recipes.values():
            if x in utensil:
                for y in utensil[x]["recipe"]:
                    if y not in out:
                        out.add(y)
                        todo.append(y)
    return out


def describe(skill):
    """Plain-language meaning of a benchmark skill string, used as the Clef criterion text."""
    m = re.match(r"(\w+)\((.*)\)", skill)
    f, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
    if f == "pickup":
        return f"walk to the {args[1].replace('_', ' ')} and pick up the {args[0]}"
    return {"put_obj_in_utensil": f"put the held item into {args[0] if args else ''}",
            "place_obj_on_counter": "put the held item down on the shared counter",
            "fill_dish_with_food": f"fill the held dish with the food in {args[0] if args else ''}",
            "deliver_soup": "serve the held dish at the serving station",
            "wait": "do nothing this tick"}.get(f, f"start {f} on {args[0] if args else ''}")


class ClefAgents(LLMAgents):
    log = None  # set by runner: list collecting per-decision records
    planner_ = None  # set by runner for hierarchical mode: Planner instance shared by both agents

    def candidates(self, state):
        # recipe items plus whatever is in the kitchen now, so off-recipe mistakes can be undone
        live = {o.name for o in state.objects.values() if o.name != "soup"}
        live |= {o.state[0] for o in state.objects.values()
                 if o.name == "soup" and isinstance(getattr(o, "state", None), tuple) and isinstance(o.state[0], str)}
        items = sorted(tree_items(self.mdp, self.order) | live | {"dish"})
        utensils = [u for u in self.mdp.utensil_list]
        places = ["ingredient_dispenser", "dish_dispenser", "counter"] + utensils
        c = [f"pickup({o},{p})" for o in items for p in places]
        c += [f"put_obj_in_utensil({u})" for u in utensils]
        c += [f"{op}({u})" for op in OPS[self.actor] for u in utensils]
        c += [f"fill_dish_with_food({u})" for u in utensils]
        c += ["place_obj_on_counter()", "deliver_soup()"]
        ok = []
        saved = (self.current_ml_action, self.time_to_wait)
        for a in c:
            self.current_ml_action = a
            try:
                if "success" in self.validate_current_ml_action(state):
                    ok.append(a)
            except Exception:
                pass
        self.current_ml_action, self.time_to_wait = saved
        return ok + ["wait(1)"]

    def clef_prompt(self, state):
        lines = [f"Task: cook and serve one {self.order}. You are the {self.name}.",
                 ROLE[self.actor], "Recipe:", recipe_text(self.order),
                 "Kitchen:", self.generate_state_prompt(state)]
        hist = [x["action"] for x in self.teammate_ml_actions[-5:]]
        if hist:
            lines.append(f"{self.teammate.name} recently did: {', '.join(hist)}")
        return "\n".join(lines)

    def compact_state(self, state, goals):
        """Short JSON state for hierarchical mode: only fields that bear on the next skill."""
        me, mate = state.players[self.agent_index], state.players[1 - self.agent_index]
        held = lambda p: p.held_object.name if p.held_object else None
        ut = {u: (s["status"] if not (s["contents"] or s["product"]) else
                  f"{s['status']}: {s['product'] or ', '.join(s['contents'])}")
              for u, s in self.planner_.utensils(state).items()}
        counters = set(self.mdp.get_counter_locations())
        return {"you": self.name.lower(), "your_goal": goals[self.agent_index],
                "teammate_goal": goals[1 - self.agent_index],
                "you_hold": held(me), "teammate_holds": held(mate), "utensils": ut,
                "counter_items": sorted(o.name for o in state.objects.values() if o.position in counters)}

    def generate_ml_action(self, state):
        if self.actor == "chef" and self.order not in self.recipe:
            self.load_recipe()
        opts = self.candidates(state)
        goal = None
        if self.planner_ is not None:
            goals = self.planner_.goals(state)
            goal = goals[self.agent_index]
            if goal is None:  # planner has no work for this agent: idle instead of interfering
                opts = ["wait(1)"]
            elif not goal.lower().startswith("wait") and len(opts) > 1:
                opts = [o for o in opts if o != "wait(1)"]  # mask idling when there is work
            prompt = json.dumps(self.compact_state(state, goals))
            q = {"next_skill": {"type": "choice",
                                "instructions": "Pick the skill that moves you toward your_goal.",
                                "criteria": {o: describe(o) for o in opts}}}
        else:
            prompt = self.clef_prompt(state)
            q = {"next_skill": {"type": "choice",
                                "instructions": f"Pick the {self.name}'s next skill that best advances the order.",
                                "criteria": {o: o for o in opts}}}
        t0, probs, err = time.time(), None, None
        if len(opts) == 1:
            probs = {opts[0]: 1.0}
        elif os.environ.get("WORKER") == "gemma":  # baseline: Gemma picks the skill instead of Clef
            try:
                menu = "\n".join(f"- {o}: {describe(o)}" for o in opts)
                r = requests.post(os.environ.get("GEMMA_URL", "http://127.0.0.1:8082") + "/v1/chat/completions",
                                  timeout=60, json={"temperature": 0, "max_tokens": 40,
                                                    "chat_template_kwargs": {"enable_thinking": False},
                                                    "messages": [{"role": "user", "content":
                                                        f"State: {prompt}\nPick the skill that moves you toward your_goal.\n"
                                                        f"Options:\n{menu}\nReply with only the option, exactly as written."}]})
                text = r.json()["choices"][0]["message"]["content"].strip().strip("`").replace(" ", "")
                pick = next((o for o in opts if o.replace(" ", "") in text), None)
                probs = {o: float(o == pick) for o in opts} if pick else {"wait(1)": 1.0}
                if not pick:
                    err = f"unparsed: {text[:60]}"
            except Exception as e:
                err, probs = repr(e), {"wait(1)": 1.0}
        elif os.environ.get("RANDOM_WORKER"):  # ablation: same planner, uniform random skill instead of Clef
            import random
            probs = {o: 1.0 / len(opts) for o in opts}
            probs[random.Random(hash((state.timestep, self.agent_index))).choice(opts)] += 1e-6
        else:
            try:
                r = requests.post(f"{URL}/v1/systemone", timeout=60,
                                  json={"model": "clef-flash", "state": prompt, "questions": q})
                r.raise_for_status()
                probs = r.json()["answers"]["next_skill"]["probabilities"]
            except Exception as e:
                err = repr(e)
                probs = {"wait(1)": 1.0}
        choice = max(probs, key=probs.get)
        if self.log is not None:
            self.log.append({"t": state.timestep, "agent": self.agent_index, "actor": self.actor,
                             "goal": goal, "prompt_chars": len(prompt),
                             "options": opts, "probs": probs, "choice": choice,
                             "latency_s": time.time() - t0, "err": err})
        m = re.search(r"wait\((\d+)\)", choice)
        if m:
            self.time_to_wait = int(m.group(1))
        self.current_ml_action_steps = 0
        return choice
