"""Strategic layer: a non-Clef planner that turns the order's recipe tree into one goal per agent.

Feudal-style split. The planner (manager) knows the recipe chain, for example
potato -> cut -> potato_slices -> bake -> baked_potato_slices -> serve, and who can
reach which utensil. Each tick it reads the game state and returns the current
goal for each agent. Clef (worker) picks the concrete skill that reaches the goal.

Goals are recomputed from state every tick, not replayed from a script, so the plan
recovers when a worker does something unexpected. Layout knowledge (which side of the
wall each station is on) is specific to Collab-Overcooked's `new_env` layout.
"""
import re

CHEF, ASSIST = 0, 1
SIDE_X = 2  # middle wall column; x < 2 is the assistant side, x > 2 the chef side


def side_of(pos):
    x = pos[0]
    return None if x == SIDE_X else (ASSIST if x < SIDE_X else CHEF)


class Planner:
    def __init__(self, mdp, order):
        self.mdp, self.order = mdp, order
        self.producer = {}  # item -> (utensil instance, inputs)
        for u in mdp.utensil_list:
            for item, r in mdp.recipes.get(re.sub(r"\d+", "", u), {}).items():
                self.producer.setdefault(item, (u, list(r["recipe"])))
        self.need_dish = mdp.need_dish.get(order, 0) == 1

    # ---- reading the state ----
    def utensils(self, state):
        status = self.mdp.get_utensil_states(state)
        out = {}
        for u, d in self.mdp.utensil_state_dict.items():
            st = next((k for k, v in status.items() if u in v), "empty")
            soup = d["soup"]
            if soup is None:
                out[u] = {"status": "empty", "contents": [], "product": None}
            elif isinstance(soup.state[0], str):
                out[u] = {"status": st, "contents": [], "product": soup.state[0]}
            else:
                out[u] = {"status": st, "contents": list(soup.state[0]), "product": None}
        return out

    def locate(self, state, item):
        """Where `item` is now: ('held', agent) | ('counter', side or None) | ('utensil', u, status) | None."""
        for i, p in enumerate(state.players):
            if p.held_object and p.held_object.name == item:
                return ("held", i)
        counters = set(self.mdp.get_counter_locations())
        for o in state.objects.values():
            if o.name == item and o.position in counters:
                return ("counter", side_of(o.position))
        for u, s in self.utensils(state).items():
            if s["product"] == item:
                return ("utensil", u, s["status"])
        return None

    def u_side(self, u):
        return side_of(self.mdp.utensil_state_dict[u]["position"])

    # ---- building goals ----
    def goals(self, state):
        """Return {agent: goal_text or None}. None means no useful work right now."""
        g = {CHEF: None, ASSIST: None}

        def add(agent, text):
            if g[agent] is None:
                g[agent] = text

        self._need(state, self.order, CHEF, add, final=True)
        return g

    def _bring(self, state, item, loc, to_side, add, why):
        """Item exists at `loc`; make sure it ends up usable by agent on `to_side`."""
        kind = loc[0]
        if kind == "held":
            holder = loc[1]
            if holder == to_side:
                return True
            add(holder, f"put the {item} on the shared counter so the {name(to_side)} can {why}")
            return False
        if kind == "counter":
            if loc[1] in (None, to_side):
                add(to_side, f"pick up the {item} from the counter to {why}")
                return False
            add(1 - to_side, f"pick up the {item} and move it to the shared counter")
            return False
        # in a utensil
        u, status = loc[1], loc[2]
        owner = self.u_side(u)
        if status == "cooking":
            add(owner, f"wait: {u} is still making {item}")
            return False
        if owner == to_side:
            add(to_side, f"take the {item} out of {u} to {why}")
        else:
            add(owner, f"take the {item} out of {u} and put it on the shared counter for the {name(to_side)}")
        return False

    def _need(self, state, item, side, add, final=False):
        loc = self.locate(state, item)
        if final and loc is not None:
            if loc[0] == "held" and loc[1] == CHEF:
                add(CHEF, f"serve the {item} now")
                return
            if loc[0] == "utensil" and loc[2] == "ready" and self.need_dish:
                chef_dish = self.locate(state, "dish")
                if chef_dish and chef_dish[0] == "held" and chef_dish[1] == CHEF:
                    add(CHEF, f"fill the dish with {item} from {loc[1]}")
                    return
                if chef_dish:
                    self._bring(state, "dish", chef_dish, CHEF, add, "plate the food")
                else:
                    add(ASSIST, "get a dish from the dish dispenser and put it on the shared counter")
                return
        if loc is not None:
            self._bring(state, item, loc, side, add, "serve it" if final else "use it")
            return
        if item not in self.producer:  # raw ingredient, only the assistant reaches the dispenser
            add(ASSIST, f"get a {item} from the ingredient dispenser" +
                ("" if side == ASSIST else " and put it on the shared counter for the chef"))
            return
        u, inputs = self.producer[item]
        owner = self.u_side(u)
        us = self.utensils(state)[u]
        op = self.mdp.utensils[re.sub(r"\d+", "", u)]["operation"]
        if us["status"] == "cooking" and us["product"] == item:
            add(owner, f"wait: {u} is making {item}")
            return
        missing = [x for x in inputs if x not in us["contents"]]
        if not missing:
            add(owner, f"start {op} on {u} to make {item}")
            return
        for x in missing:
            xl = self.locate(state, x)
            if xl and xl[0] == "held" and xl[1] == owner:
                add(owner, f"put the {x} into {u} (to make {item})")
            elif xl is not None:
                self._bring(state, x, xl, owner, add, f"put it into {u}")
            else:
                self._need(state, x, owner, add)


def name(agent):
    return "chef" if agent == CHEF else "assistant"
