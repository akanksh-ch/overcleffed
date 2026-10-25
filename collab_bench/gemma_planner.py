"""Strategic layer as an LLM: Gemma 4 sets one typed goal per agent; Clef still picks the skills.

Same interface as planner.Planner (`goals(state)`, `utensils(state)`), so the Clef worker is
unchanged. Gemma output is typed, in the spirit of Jev/Clef decision APIs:
1. Schema: llama-server constrains decoding to a JSON schema. Each agent's `station` enum only
   lists stations that agent can reach, so unreachable goals cannot be written.
2. Preconditions: code checks each goal against the state (for example "start" needs a loaded
   utensil, "wait" needs a working one). A failing goal is sent back with the error, up to
   RETRIES times; failures retain valid unfinished work or use the deterministic planner.
Gemma is called when the kitchen changes, or after STALL ticks without change.
"""
import json
import os
import re
import time

import requests

from planner import Planner

STALL = 5
RETRIES = 2
URL = os.environ.get("GEMMA_URL", "http://127.0.0.1:8082")
MODEL = os.environ.get("GEMMA_MODEL")  # e.g. gemma-4-26b-a4b-it; unset = local llama-server
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
RULES = """Kitchen rules:
- Two agents. The chef reaches pot0, oven0, the serving station, and the shared counter.
- The assistant reaches the ingredient dispenser, the dish dispenser, chopping_board0, blender0, and the shared counter.
- Items cross sides only by one agent placing them on the shared counter and the other taking them.
- An agent holds one item at a time. Put the inputs into a utensil, then "start" it. It works for 3 ticks, then the product must be taken out.
- A utensil that is "loaded, not started" does nothing until an agent starts it.
- Some orders must be plated: the chef holds a dish and fills it from the utensil, then serves. Only the chef serves."""
ACTIONS = {
    0: ["take_from_counter", "put_into", "start", "take_out", "place_on_counter", "fill_dish", "serve", "wait", "idle"],
    1: ["get_from_dispenser", "take_from_counter", "put_into", "start", "take_out", "place_on_counter", "wait", "idle"],
}
STATIONS = {0: ["pot0", "oven0"], 1: ["chopping_board0", "blender0"]}
LABEL = {"full": "loaded, not started", "partially_full": "partly loaded", "cooking": "working",
         "ready": "done, product waiting to be taken out", "empty": "empty", "wrong": "wrong items"}


def recipe_text(order):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "recipe")
    for f in os.listdir(d):
        if order in f.lower():
            return open(os.path.join(d, f)).read().strip()
    return ""


def sentence(g):
    """Typed goal -> plain sentence for the Clef worker."""
    a, it, st = g["action"], g.get("item"), g.get("station")
    return {"get_from_dispenser": f"get a {it} from the {'dish' if it == 'dish' else 'ingredient'} dispenser",
            "take_from_counter": f"pick up the {it} from the shared counter",
            "put_into": f"put the {it} into {st}",
            "start": f"start {st}",
            "take_out": f"take the {it} out of {st}",
            "place_on_counter": f"put the {it} on the shared counter",
            "fill_dish": f"fill the dish with food from {st}",
            "serve": f"serve the {it}",
            "wait": f"wait: {st} is working"}.get(a)


class GemmaPlanner(Planner):
    def __init__(self, mdp, order):
        super().__init__(mdp, order)
        self.recipe = recipe_text(order)
        items = sorted(self._tree(order) | {"dish"})
        agent = lambda i: {"type": "object", "required": ["action", "item", "station"], "properties": {
            "action": {"enum": ACTIONS[i]},
            "item": {"enum": items + [None]},
            "station": {"enum": STATIONS[i] + [None]}}}
        # reasoning comes first so the model thinks before the typed fields are decoded
        self.schema = {"type": "object", "required": ["reasoning", "chef", "assistant"],
                       "properties": {"reasoning": {"type": "string", "maxLength": 250},
                                      "chef": agent(0), "assistant": agent(1)}}
        self.sig, self.since, self.typed, self.cached = None, 0, {0: None, 1: None}, {0: None, 1: None}
        self.calls = []  # per Gemma call: t, latency, raw goals, validation errors, final goals

    def _tree(self, item):
        out, todo = {item}, [item]
        while todo:
            x = todo.pop()
            if x in self.producer:
                for y in self.producer[x][1]:
                    if y not in out:
                        out.add(y)
                        todo.append(y)
        return out

    def snapshot(self, state):
        held = [p.held_object.name if p.held_object else None for p in state.players]
        counters = set(self.mdp.get_counter_locations())
        ut = {u: {k: (LABEL.get(v, v) if k == "status" else v) for k, v in s.items() if v}
              for u, s in self.utensils(state).items()}
        items = sorted(o.name for o in state.objects.values() if o.position in counters)
        return {"chef_holds": held[0], "assistant_holds": held[1], "utensils": ut, "shared_counter": items}

    def check(self, i, g, state):
        try:
            return self._check(i, g, state)
        except (KeyError, TypeError, AttributeError) as e:  # model broke the schema
            return f"invalid goal {json.dumps(g)[:80]}: {e!r}"

    def completed(self, i, g, state):
        """Observe effects before checking the action's now-vanished preconditions."""
        if not g:
            return False
        a, it, st = g['action'], g.get('item'), g.get('station')
        held = state.players[i].held_object
        held = held.name if held else None
        ut = self.utensils(state).get(st, {})
        if a in ('get_from_dispenser', 'take_from_counter', 'take_out'):
            return held == it
        if a == 'place_on_counter':
            return held != it and self.locate(state, it) is not None
        if a == 'put_into':
            return held != it and (it in ut.get('contents', []) or ut.get('status') in ('cooking', 'ready'))
        if a == 'start':
            return ut.get('status') in ('cooking', 'ready')
        if a == 'wait':
            return ut.get('status') == 'ready'
        if a == 'fill_dish':
            return held == self.order
        # Serving needs delivery evidence, not merely empty hands.
        return False

    def _check(self, i, g, state):
        """Precondition check for agent i's typed goal. Returns an error string or None."""
        if not isinstance(g, dict) or not {'action', 'item', 'station'} <= g.keys():
            return 'goal must contain action, item and station'
        a, it, st = g['action'], g['item'], g['station']
        if a not in ACTIONS[i] or it not in self._tree(self.order) | {'dish', None} or st not in STATIONS[i] + [None]:
            return 'goal fields violate agent schema'
        held = state.players[i].held_object.name if state.players[i].held_object else None
        ut = self.utensils(state)
        counters = set(self.mdp.get_counter_locations())
        on_counter = {o.name for o in state.objects.values() if o.position in counters}
        if a == "idle":
            return None
        if a in ("put_into", "start", "take_out", "fill_dish", "wait") and not st:
            return f"{a} needs a station"
        if a in ("put_into", "start", "take_out", "wait") and st not in ut:
            return f"{st} is not a utensil"
        if a in ("get_from_dispenser", "take_from_counter", "put_into", "take_out", "place_on_counter", "serve") and not it:
            return f"{a} needs an item"
        if a in ("get_from_dispenser", "take_from_counter", "take_out") and held:
            return f"agent already holds {held}"
        if a in ("put_into", "place_on_counter", "serve") and held != it:
            return f"agent holds {held}, not {it}"
        if a == "get_from_dispenser" and it != "dish" and it in self.producer:
            return f"{it} is not raw; it must be made in {self.producer[it][0]}"
        if a == "take_from_counter" and it not in on_counter:
            return f"there is no {it} on the shared counter"
        if a == "put_into":
            uses = [x for x in self._tree(self.order) if x in self.producer
                    and self.producer[x][0] == st and it in self.producer[x][1]]
            if not uses:
                return f"the recipe never puts {it} into {st}"
        if a == "start" and ut[st]["status"] not in ("full", "partially_full"):
            return f"{st} is {LABEL.get(ut[st]['status'])}, it cannot be started"
        if a == "wait" and ut[st]["status"] != "cooking":
            return f"{st} is {LABEL.get(ut[st]['status'])}, not working; waiting will not change anything"
        if a == "take_out" and ut[st]["product"] != it:
            return f"{st} does not hold a finished {it}"
        if a == "fill_dish" and held != "dish":
            return "the chef must hold a dish to fill it"
        if a == "serve" and it != self.order:
            return f"{it} is not the order"
        return None

    def ask(self, prompt):
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": {"name": "goals", "schema": self.schema}}}
        if MODEL:  # hosted Gemma via the Gemini API's OpenAI-compatible endpoint
            for wait in (2, 4, 8, 16, 0):  # back off on 429 rate limits
                r = requests.post(GEMINI_URL, timeout=60, json={**body, "model": MODEL},
                                  headers={"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"})
                if r.status_code != 429 or not wait:
                    break
                time.sleep(wait)
        else:  # local llama-server
            r = requests.post(f"{URL}/v1/chat/completions", timeout=60,
                              json={**body, "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])

    def goals(self, state):
        retired = {}
        for i, g in self.typed.items():
            if g and (self.completed(i, g, state) or self.check(i, g, state)):
                retired[i] = 'completed' if self.completed(i, g, state) else 'invalid'
                self.typed[i], self.cached[i] = None, None
        snap = self.snapshot(state)
        sig = json.dumps(snap, sort_keys=True)
        if not retired and sig == self.sig and state.timestep - self.since < STALL:
            return dict(self.cached)
        stalled = sig == self.sig
        self.sig, self.since = sig, state.timestep
        base = (f"{RULES}\n\nOrder: {self.order}\nRecipe:\n{self.recipe}\n\n"
                f"Current kitchen (tick {state.timestep}):\n{json.dumps(snap)}\n"
                f"Previous goals: {json.dumps({'chef': self.typed[0], 'assistant': self.typed[1]})}\n"
                + (f"The kitchen has not changed for {STALL} ticks, so the previous goals are not working.\n"
                   if stalled else "")
                + "\nIn reasoning, say which recipe step is next, where each needed item is now, and who can reach it. "
                  "Then give each agent its single next goal. Use action idle if it has nothing useful to do.")
        t0, errs, raw, feedback = time.time(), [], None, ""
        for _ in range(RETRIES + 1):
            try:
                raw = self.ask(base + feedback)
                if (not isinstance(raw, dict) or not all(k in raw for k in ('chef', 'assistant'))
                        or not isinstance(raw.get('reasoning'), str)):
                    raise ValueError('response must contain reasoning, chef and assistant goals')

            except Exception as e:
                errs.append(repr(e))
                raw = None
                break
            bad = {i: self.check(i, raw[k], state) for i, k in ((0, "chef"), (1, "assistant"))}
            if not any(bad.values()):
                break
            errs.append({k: bad[i] for i, k in ((0, "chef"), (1, "assistant")) if bad[i]})
            feedback = "\n\nYour last answer was rejected:\n" + "\n".join(
                f"- {k} goal {json.dumps(raw[k])}: {bad[i]}" for i, k in ((0, "chef"), (1, "assistant")) if bad[i])
        fallback, sources, failed = None, {}, False
        for i, k in ((0, 'chef'), (1, 'assistant')):
            g = raw.get(k) if isinstance(raw, dict) else None
            if g is not None and self.check(i, g, state) is None:
                self.typed[i] = g
                self.cached[i] = sentence(g)
                sources[i] = 'intentional_idle' if g['action'] == 'idle' else 'model'
            else:
                failed = True
                if self.typed[i] is not None and self.typed[i]['action'] != 'idle':
                    sources[i] = 'retained'
                else:
                    if fallback is None:
                        fallback = super().goals(state)
                    self.typed[i] = None
                    self.cached[i] = fallback[i]
                    sources[i] = 'fallback'
        self.calls.append({'t': state.timestep, 'latency_s': time.time() - t0, 'raw': raw,
                           'errors': errs, 'goals': dict(self.cached), 'sources': sources,
                           'retired': retired, 'err': (errs[-1] if errs else 'invalid response') if failed else None})
        return dict(self.cached)
