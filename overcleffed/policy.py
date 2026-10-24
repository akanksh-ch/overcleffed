"""Policies: Clef (one call, two choice questions), mock, and greedy baseline."""
import json
import math
import random
import sys

import requests
from overcooked_ai_py.mdp.actions import Action

# ---- Action space abstraction (swap RawActions for a macro-action space later) ----
class RawActions:
    """Maps option names <-> overcooked actions. Macro spaces implement the same API:
    `options` (name->description) and `to_env_action(name, state, chef_idx)`."""
    options = {
        "up": "move one tile up (y-1); if blocked, only turn to face up",
        "down": "move one tile down (y+1); if blocked, only turn to face down",
        "left": "move one tile left (x-1); if blocked, only turn to face left",
        "right": "move one tile right (x+1); if blocked, only turn to face right",
        "stay": "do nothing this tick",
        "interact": "use the tile the chef faces: grab onion/dish, drop/pick on counter, put onion in pot, plunge dish into cooked soup, deliver soup to serving",
    }
    _map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
            "stay": (0, 0), "interact": "interact"}
    stay = "stay"

    def to_env_action(self, name, state=None, chef_idx=0):
        return self._map[name]


def build_questions(space):
    return {
        f"chef_{i}_action": {
            "type": "choice",
            "instructions": (f"Choose the best next action for chef {i} in a cooperative Overcooked kitchen. "
                             "Goal: deliver as many soups as possible (onions x3 -> pot -> cook -> dish -> serve). "
                             "Avoid blocking the other chef."),
            "criteria": space.options,
        } for i in (0, 1)
    }


def parse_choice(ans, names):
    """Tolerant: returns {name: prob} normalised, or None."""
    def from_obj(o):
        if isinstance(o, dict):
            if all(n in o for n in names) and all(isinstance(o[n], (int, float)) for n in names):
                return {n: float(o[n]) for n in names}
            for k in ("probabilities", "probs", "options", "distribution", "scores", "choices", "p"):
                if k in o:
                    r = from_obj(o[k])
                    if r:
                        return r
            for v in o.values():
                if isinstance(v, (dict, list)):
                    r = from_obj(v)
                    if r:
                        return r
        elif isinstance(o, list):
            if len(o) == len(names) and all(isinstance(x, (int, float)) for x in o):
                return dict(zip(names, map(float, o)))
            d = {}
            for it in o:
                if isinstance(it, dict):
                    n = it.get("option") or it.get("name") or it.get("label") or it.get("id")
                    p = next((it[k] for k in ("probability", "prob", "p", "score") if k in it), None)
                    if n in names and p is not None:
                        d[n] = float(p)
            if d:
                return {n: d.get(n, 0.0) for n in names}
        return None
    r = from_obj(ans)
    if not r:
        return None
    s = sum(max(v, 0) for v in r.values()) or 1.0
    return {n: max(v, 0) / s for n, v in r.items()}


def decode(probs, mode="argmax", temperature=1.0, rng=random):
    if mode == "argmax":
        return max(probs, key=probs.get)
    names = list(probs)
    w = [math.exp(math.log(max(probs[n], 1e-9)) / max(temperature, 1e-6)) for n in names]
    return rng.choices(names, weights=w)[0]


class Policy:
    """act(payload, state) -> (names[2], probs[2] or None, info)"""
    def __init__(self, space=None):
        self.space = space or RawActions()
    def reset(self, env): pass


class ClefPolicy(Policy):
    def __init__(self, url="http://localhost:8080", decode_mode="argmax", temperature=1.0,
                 timeout=60, space=None, log_raw=sys.stderr):
        super().__init__(space)
        self.url, self.mode, self.temp, self.timeout = url, decode_mode, temperature, timeout
        self.names = list(self.space.options)
        self.questions = build_questions(self.space)
        self._raw_logged, self.log_raw = False, log_raw
        self.sess = requests.Session()

    def act(self, payload, state):
        body = {"model": "clef-flash", "state": payload, "questions": self.questions}
        r = self.sess.post(f"{self.url}/v1/systemone", json=body, timeout=self.timeout)
        r.raise_for_status()
        raw = r.json()
        if not self._raw_logged:
            self._raw_logged = True
            print("RAW CLEF RESPONSE:", json.dumps(raw)[:3000], file=self.log_raw)
        answers = raw.get("response", raw).get("answers", raw.get("answers", {}))
        names, probs = [], []
        for i in (0, 1):
            p = parse_choice(answers.get(f"chef_{i}_action"), self.names)
            if p is None:
                p = {n: float(n == self.space.stay) for n in self.names}
            probs.append(p)
            names.append(decode(p, self.mode, self.temp))
        return names, probs, {}


class MockPolicy(Policy):
    """Random policy biased towards interact (so soups sometimes happen)."""
    def __init__(self, seed=0, space=None):
        super().__init__(space)
        self.rng = random.Random(seed)
    def act(self, payload, state):
        names = list(self.space.options)
        w = [3, 3, 3, 3, 1, 3]
        out = [self.rng.choices(names, weights=w)[0] for _ in (0, 1)]
        return out, None, meta


class GreedyPolicy(Policy):
    """overcooked_ai GreedyHumanModel for both chefs (baseline)."""
    def reset(self, env):
        from overcooked_ai_py.agents.agent import GreedyHumanModel
        from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
        mlam = MediumLevelActionManager.from_pickle_or_compute(env.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.agents = [GreedyHumanModel(mlam) for _ in (0, 1)]
        for i, a in enumerate(self.agents):
            a.set_mdp(env.mdp); a.reset(); a.set_agent_index(i)
    def act(self, payload, state):
        acts = [a.action(state)[0] for a in self.agents]
        inv = {v: k for k, v in RawActions._map.items()}
        return [inv[a] for a in acts], None, {}


# ---- Macro-action (subgoal) policies ----
class MacroPolicy(Policy):
    """Chefs follow subgoal plans; the chooser (Clef or mock) is queried only when a chef needs one.
    act() returns raw action names, so runner/replay stay unchanged."""
    def __init__(self, chooser, history_k=2, roles=False):
        super().__init__(RawActions())
        self.chooser, self.k, self.roles = chooser, history_k, roles
        self.macro_space = None

    @property
    def options(self):
        from overcleffed.macro import SUBGOALS
        return SUBGOALS

    def reset(self, env):
        from overcleffed.macro import MacroActions
        self.env = env
        self.ms = MacroActions(env)
        self.ms.reset()
        self.hist = [[], []]
        self.score = 0
        self.horizon = env.horizon

    def act(self, payload, state):
        from overcleffed.env import compact_prompt
        ms = self.ms
        asked = [i for i in (0, 1) if ms.needs_subgoal(state, i)]
        info = {"clef_called": False, "subgoal_probs": [None, None], "input_tokens": None, "confidence": None}
        if asked:
            opts = {i: ms.options_for(state, i) for i in asked}
            keep = [(ms.plans[j]["sub"], 0) if ms.plans[j] else None for j in (0, 1)]
            prompt = compact_prompt(self.env.mdp, state, payload.get("score", 0), self.horizon, keep,
                                    [h[-self.k:] if self.k else [] for h in self.hist], self.roles)
            forced = {i: next(iter(o)) for i, o in opts.items() if len(o) == 1}  # nothing to choose
            ask = {i: o for i, o in opts.items() if i not in forced}
            choice, probs, meta = dict(forced), {}, {}
            if ask:
                c, probs, meta = self.chooser(prompt, ask, state)
                choice.update(c)
            hc = [heuristic_choice(ask[i], i) if i in ask else None for i in (0, 1)]
            info["heuristic_choice"] = hc
            info["agree"] = [choice[i] == hc[i] if i in ask else None for i in (0, 1)]
            info["conf_by_chef"] = [(meta.get("conf_by_chef") or {}).get(i) for i in (0, 1)]
            for i in asked:
                ms.assign(i, choice[i], state)
                self.hist[i].append(choice[i])
                self.hist[i] = self.hist[i][-max(self.k, 1):]
                info["subgoal_probs"][i] = probs.get(i) if probs else None
            info["clef_called"] = bool(ask) and meta.get("called", False)
            info["input_tokens"] = meta.get("input_tokens")
            info["confidence"] = meta.get("confidence")
        info["subgoal"] = [ms.plans[i]["sub"] for i in (0, 1)]
        info["asked"] = asked
        inv = {v: k for k, v in RawActions._map.items()}
        names = []
        for i in (0, 1):
            a = ms.step_action(state, i)
            names.append(inv[a])
        return names, None, info


def heuristic_choice(opts, i):
    """Rule-based reference choice (chef 0 favours onions, chef 1 dishes)."""
    order = ["start_cooking", "deliver_soup", "pickup_soup", "put_onion_in_pot",
             "get_onion", "get_dish"] if i == 0 else \
            ["start_cooking", "deliver_soup", "pickup_soup", "put_onion_in_pot", "get_dish", "get_onion"]
    return next((n for n in order if n in opts), next(iter(opts)))


def mock_chooser(seed=0, heuristic=True):
    rng = random.Random(seed)
    def choose(prompt, opts, state):
        meta = {"called": True}
        if not heuristic:
            return {i: rng.choice(list(o)) for i, o in opts.items()}, None, meta
        return {i: heuristic_choice(o, i) for i, o in opts.items()}, None, meta
    return choose


def clef_chooser(url, decode_mode="argmax", temperature=1.0, timeout=120, log=sys.stderr,
                 question_mode="choice"):
    sess, seen = requests.Session(), [False]
    from overcleffed.macro import SUBGOALS, WAIT_TICKS
    def choose(prompt, opts, state):
        qs = {}
        if question_mode == "noul":
            for i, o in opts.items():
                for n, c in o.items():
                    qs[f"chef_{i}_{n}"] = {"type": "noul", "instructions":
                        f"Chef {i}: does {n} ({SUBGOALS[n]}; {c} steps) move the team closest to delivering soup right now?"}
        for i, o in ({} if question_mode == "noul" else opts).items():
            qs[f"chef_{i}_subgoal"] = {
                "type": "choice",
                "instructions": (f"Pick chef {i}'s next subgoal. Goal: serve many soups "
                                 "(3 onions in pot, cook, dish, soup, serve). Keep both chefs busy, avoid blocking."),
                "criteria": {n: (f"{SUBGOALS[n]} ({c} steps)" if n != "wait" else f"idle {WAIT_TICKS} ticks")
                             for n, c in o.items()}}
        r = sess.post(f"{url}/v1/systemone", json={"model": "clef-flash", "state": prompt, "questions": qs},
                      timeout=timeout)
        r.raise_for_status()
        raw = r.json()
        if not seen[0]:
            seen[0] = True
            print("RAW CLEF RESPONSE:", json.dumps(raw)[:2000], file=log)
        answers = raw.get("response", raw).get("answers", raw.get("answers", {}))
        usage = raw.get("usage") or raw.get("response", {}).get("usage") or {}
        choice, probs, confs, cbc = {}, {}, [], {}
        if question_mode == "noul":
            for i, o in opts.items():
                v = {n: float((answers.get(f"chef_{i}_{n}") or {}).get("noul", 0.0)) for n in o}
                s = sum(v.values()) or 1.0
                probs[i] = v  # raw noul scores per subgoal
                choice[i] = decode({n: x / s for n, x in v.items()}, decode_mode, temperature)
                confs.append(v[choice[i]])
                cbc[i] = v[choice[i]]
            return choice, probs, {"called": True, "input_tokens": usage.get("input_tokens"),
                                   "confidence": sum(confs) / len(confs), "conf_by_chef": cbc}
        for i, o in opts.items():
            a = answers.get(f"chef_{i}_subgoal")
            p = parse_choice(a, list(o)) or {n: float(n == "wait") for n in o}
            probs[i] = p
            choice[i] = decode(p, decode_mode, temperature)
            if isinstance(a, dict) and "confidence" in a:
                confs.append(a["confidence"])
                cbc[i] = a["confidence"]
        return choice, probs, {"called": True, "input_tokens": usage.get("input_tokens"),
                               "confidence": sum(confs) / len(confs) if confs else None, "conf_by_chef": cbc}
    return choose
