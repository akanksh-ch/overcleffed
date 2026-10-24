"""Macro-action (subgoal) space: per-chef plans executed with overcooked_ai planners."""
import os

from overcooked_ai_py.planning.planners import MediumLevelActionManager

SUBGOALS = {
    "get_onion": "grab an onion from dispenser/counter",
    "start_cooking": "with empty hands: start cooking a full pot (needs 3 onions)",
    "put_onion_in_pot": "walk to a pot and drop the held onion in",
    "get_dish": "grab a dish from dispenser/counter",
    "pickup_soup": "with a dish: scoop soup from a cooking/ready pot",
    "deliver_soup": "carry held soup to serving window (+20)",
    "place_on_counter": "drop held item on a free counter",
    "wait": "stay put for a few ticks",
}
WAIT_TICKS = 3
STUCK_LIMIT = 3
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "mlam_cache")


def load_mlam(mdp):
    cl = mdp.get_counter_locations()
    params = {"start_orientations": False, "wait_allowed": False, "counter_goals": cl,
              "counter_drop": cl, "counter_pickup": cl, "same_motion_goals": True}
    os.makedirs(CACHE_DIR, exist_ok=True)
    fn = os.path.join(CACHE_DIR, f"{mdp.layout_name}_am.pkl")
    return MediumLevelActionManager.from_pickle_or_compute(mdp, params, custom_filename=fn)


def h_now(p):
    return p.held_object.name if p.held_object else None


def held(player):
    return player.held_object.name if player.held_object else None


class MacroActions:
    """Subgoal space. `options_for(state, i)` lists feasible subgoals (with step counts)."""
    stay = "wait"
    options = SUBGOALS

    def __init__(self, env):
        self.mdp = env.mdp
        self.mlam = load_mlam(self.mdp)
        self.mp = self.mlam.motion_planner
        self.cl = self.mdp.get_counter_locations()

    def reset(self):
        self.plans = [None, None]  # {"sub","goal","wait","stuck","last"}
        self.banned = [None, None]  # (sub, goal) that got stuck; avoided when alternatives exist

    # ---- feasibility ----
    def goals(self, sub, state, i):
        """Valid (pos, orientation) motion goals for subgoal `sub`, chef i."""
        m, mdp, p = self.mlam, self.mdp, state.players[i]
        h = held(p)
        ps = mdp.get_pot_states(state)
        if sub == "get_onion" and h is None:
            g = m.pickup_onion_actions(mdp.get_counter_objects_dict(state, self.cl))
        elif sub == "get_dish" and h is None:
            g = m.pickup_dish_actions(mdp.get_counter_objects_dict(state, self.cl))
        elif sub == "put_onion_in_pot" and h == "onion":
            g = m.put_onion_in_pot_actions(ps)
        elif sub == "start_cooking" and h is None:
            g = m._get_ml_actions_for_positions(mdp.get_full_but_not_cooking_pots(ps))
        elif sub == "pickup_soup" and h == "dish":
            g = m.pickup_soup_with_dish_actions(ps, only_nearly_ready=True)
        elif sub == "deliver_soup" and h == "soup":
            g = m.deliver_soup_actions()
        elif sub == "place_on_counter" and h is not None:
            g = m.place_obj_on_counter_actions(state)
        else:
            return []
        return [x for x in g if self.mp.is_valid_motion_start_goal_pair(p.pos_and_or, x)]

    def best_goal(self, sub, state, i):
        """Nearest goal, preferring tiles the partner does not occupy."""
        gs = self.goals(sub, state, i)
        if not gs:
            return None, None
        other = state.players[1 - i].position
        start = state.players[i].pos_and_or
        free = [g for g in gs if g[0] != other] or gs
        b = self.banned[i]
        if b and b[0] == sub:
            free = [g for g in free if g != b[1]] or free
        g = min(free, key=lambda g: self.mp.get_plan(start, g)[2])
        return g, self.mp.get_plan(start, g)[2]

    def options_for(self, state, i):
        out = {"wait": 0}
        for sub in SUBGOALS:
            if sub == "wait":
                continue
            g, c = self.best_goal(sub, state, i)
            if g is not None:
                out[sub] = c
        return out

    # ---- execution ----
    def assign(self, i, sub, state):
        g, _ = self.best_goal(sub, state, i) if sub != "wait" else (None, 0)
        self.plans[i] = {"sub": sub, "goal": g, "wait": WAIT_TICKS, "stuck": 0,
                         "start_held": held(state.players[i]), "last": None}

    def needs_subgoal(self, state, i):
        pl = self.plans[i]
        if pl is None:
            return True
        sub, p = pl["sub"], state.players[i]
        if sub == "wait":
            return pl["wait"] <= 0
        if pl["goal"] is None:
            return True
        if pl["stuck"] >= STUCK_LIMIT:
            self.banned[i] = (sub, pl["goal"])
            if i == 1:  # yield to chef 0: wait, then re-ask
                self.plans[i] = {"sub": "wait", "goal": None, "wait": WAIT_TICKS, "stuck": 0,
                                 "start_held": h_now(p), "last": None}
                return False
            return True
        h, h0 = held(p), pl["start_held"]
        if pl.get("did_interact") and h != h0:  # item changed -> done
            return True
        if sub in ("place_on_counter", "deliver_soup", "put_onion_in_pot") and h is None:
            return True
        if pl["goal"] not in self.goals(sub, state, i):  # target invalid now
            return True
        return False

    def step_action(self, state, i):
        """Raw env action for chef i from its current plan."""
        pl, p = self.plans[i], state.players[i]
        if pl["sub"] == "wait":
            pl["wait"] -= 1
            return (0, 0)
        cur = p.pos_and_or
        if pl["last"] is not None and pl["last"] == (cur, held(p)) and pl["last_act"] != "interact":
            pl["stuck"] += 1
        pl["last"] = (cur, held(p))
        a = self.bfs_action(state, i, pl["goal"])
        if a is None:  # goal unreachable around partner: hold and count as stuck
            pl["last_act"] = (0, 0)
            pl["stuck"] += 1
            return (0, 0)
        pl["last_act"] = a
        if a == "interact":
            pl["did_interact"] = True
            # pickup_soup with unfinished pot: keep trying (no state change)
        return a

    def bfs_action(self, state, i, goal):
        """First action of a shortest path to goal=(pos, orient) treating partner's tile as a wall."""
        from collections import deque
        mdp = self.mdp
        start = state.players[i].pos_and_or
        block = state.players[1 - i].position
        if start == goal:
            return "interact"
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        seen, q = {start: None}, deque([start])
        while q:
            cur = q.popleft()
            (x, y), _ = cur
            for d in dirs:
                nx, ny = x + d[0], y + d[1]
                if (nx, ny) != block and mdp.terrain_mtx[ny][nx] == " ":
                    nxt = ((nx, ny), d)
                else:
                    nxt = ((x, y), d)
                if nxt in seen:
                    continue
                seen[nxt] = (cur, d)
                if nxt == goal:
                    while seen[nxt][0] != start:
                        nxt = seen[nxt][0]
                    return seen[nxt][1]
                q.append(nxt)
        return None
