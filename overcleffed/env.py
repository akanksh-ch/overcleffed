"""Overcooked env construction, state serialization and history."""
from collections import deque

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

DIRS = {(0, -1): "up", (0, 1): "down", (1, 0): "right", (-1, 0): "left"}


def make_env(layout="cramped_room", horizon=400):
    mdp = OvercookedGridworld.from_layout_name(layout)
    return OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)


def layout_summary(mdp):
    return {
        "grid_size_xy": [len(mdp.terrain_mtx[0]), len(mdp.terrain_mtx)],
        "note": "x grows right, y grows down; up=y-1. Chef faces a tile and interacts with it.",
        "onion_dispensers": [list(p) for p in mdp.get_onion_dispenser_locations()],
        "dish_dispensers": [list(p) for p in mdp.get_dish_dispenser_locations()],
        "pots": [list(p) for p in mdp.get_pot_locations()],
        "serving": [list(p) for p in mdp.get_serving_locations()],
        "walls_or_counters": [[x, y] for y, row in enumerate(mdp.terrain_mtx)
                              for x, c in enumerate(row) if c == "X"],
        "soup_needs": "3 onions in a pot, cook 20 ticks, plunge dish, deliver for 20 points",
    }


def _obj(o):
    if o is None:
        return None
    d = {"name": o.name}
    if o.name == "soup":
        d["onions"] = len(o.ingredients)
        d["ready"] = bool(o.is_ready)
        if o.is_cooking:
            d["cook_left"] = max(0, o.cook_time - o._cooking_tick)
        d["cooking"] = bool(o.is_cooking)
    return d


def serialize_state(state):
    """Compact JSON-able dict of one state."""
    chefs = []
    for i, p in enumerate(state.players):
        chefs.append({"id": i, "pos": list(p.position), "facing": DIRS[p.orientation],
                      "holding": _obj(p.held_object)})
    pots, counters = [], []
    for pos, o in state.objects.items():
        entry = {"pos": list(pos), **_obj(o)}
        (pots if o.name == "soup" and False else counters).append(entry)
    return {"t": state.timestep, "chefs": chefs, "objects": counters}


def _chef_sig(c):
    h = c["holding"]
    return f'@{c["pos"][0]},{c["pos"][1]} f={c["facing"]} h={h["name"] if h else "-"}'


def _obj_sig(objs):
    out = []
    for o in sorted(objs, key=lambda o: o["pos"]):
        s = f'{o["name"]}@{o["pos"][0]},{o["pos"][1]}'
        if o["name"] == "soup":
            s += f':{o["onions"]}o' + (f',cook_left={o["cook_left"]}' if o.get("cooking") else "") + (",ready" if o["ready"] else "")
        out.append(s)
    return out


class History:
    """Last k states as terse summaries plus each chef's last actions."""

    def __init__(self, k=5):
        self.k = k
        self.states = deque(maxlen=k)
        self.actions = deque(maxlen=k)  # (a0, a1) names
        self.rewards = deque(maxlen=k)

    def push(self, ser_state, actions=None, reward=0):
        self.states.append({
            "t": ser_state["t"],
            "chef_0": _chef_sig(ser_state["chefs"][0]),
            "chef_1": _chef_sig(ser_state["chefs"][1]),
            "objects": _obj_sig(ser_state["objects"]),
        })
        self.actions.append(actions)
        self.rewards.append(reward)

    def render(self):
        """Entries pair state t with the actions taken FROM it (None for current)."""
        out = []
        for s, a, r in zip(self.states, self.actions, self.rewards):
            e = dict(s)
            if a is not None:
                e["chef_0_did"], e["chef_1_did"] = a
                e["reward_after"] = r
            out.append(e)
        return out


def build_state_payload(layout, ser_state, history, score, horizon):
    return {
        "layout": layout,
        "now": {**ser_state, "score": score, "steps_left": horizon - ser_state["t"]},
        "recent_history_oldest_first": history.render(),
    }


def _pot_line(mdp, state):
    out = []
    for pos in mdp.get_pot_locations():
        o = state.objects.get(pos)
        if o is None:
            s = "empty"
        elif o.is_ready:
            s = "READY"
        elif o.is_cooking:
            s = f"cooking {max(0, o.cook_time - o._cooking_tick)} ticks left"
        else:
            s = f"{len(o.ingredients)}/3 onions"
        out.append(f"pot{pos[0]}{pos[1]}: {s}")
    return "; ".join(out)


def compact_prompt(mdp, state, score, horizon, plans, hist):
    """Short plain-text state for macro mode. `plans[i]` = (subgoal, steps_left) or None.
    `hist[i]` = list of recent subgoal names."""
    lines = [f"t={state.timestep}/{horizon} score={score}", _pot_line(mdp, state)]
    for i, p in enumerate(state.players):
        h = p.held_object.name if p.held_object else "nothing"
        pl = plans[i]
        doing = f", doing {pl[0]}" if pl else ""
        lines.append(f"chef{i} at {p.position[0]},{p.position[1]} holds {h}{doing}")
    for i, h in enumerate(hist):
        if h:
            lines.append(f"chef{i} recent: {','.join(h)}")
    return "\n".join(lines)
