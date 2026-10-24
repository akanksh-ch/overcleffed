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
        return out, None, {}


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
