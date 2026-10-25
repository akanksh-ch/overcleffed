"""Experimental per-chef inference; only the simulation thread mutates plans."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from overcleffed.env import compact_prompt
from overcleffed.macro import held
from overcleffed.policy import MacroPolicy, RawActions


class AsyncMacroPolicy(MacroPolicy):
    def __init__(self, choosers, history_k=2, roles=False):
        super().__init__(choosers[0], history_k, roles)
        self.choosers = choosers  # separate HTTP sessions per chef
        self.pool = ThreadPoolExecutor(2)
        self.pending = [None, None]

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def reset(self, env):
        # Drain the prior episode before accepting decisions for a new one.
        for pending in self.pending:
            if pending:
                try:
                    pending[0].result()
                except Exception:
                    pass
        self.pending = [None, None]
        super().reset(env)

    def act(self, payload, state):
        ms = self.ms
        info = {"clef_called": False, "asked": [], "async_errors": [],
                "async_discarded": 0, "async_pending": [], "subgoal_probs": [None, None]}
        names = []
        inv = {v: k for k, v in RawActions._map.items()}
        for i in (0, 1):
            needs = ms.needs_subgoal(state, i)
            pending = self.pending[i]
            if pending and pending[0].done():
                self.pending[i] = None
                try:
                    choice, probs, meta = pending[0].result()
                    info["clef_called"] = meta.get("called", False) or info["clef_called"]
                    sub = choice[i]
                    # Recompute targets at acceptance, never reuse a stale path.
                    if needs and held(state.players[i]) == pending[1] and sub in ms.options_for(state, i):
                        ms.assign(i, sub, state)
                        self.hist[i].append(sub)
                        self.hist[i] = self.hist[i][-max(self.k, 1):]
                        info["subgoal_probs"][i] = probs.get(i) if probs else None
                        needs = False
                    else:
                        info["async_discarded"] += 1
                except Exception as e:
                    info["async_errors"].append(f"chef {i}: {type(e).__name__}: {e}")
            if needs and self.pending[i] is None:
                opts = ms.options_for(state, i)
                if len(opts) == 1:
                    ms.assign(i, next(iter(opts)), state)
                    needs = False
                else:
                    keep = [(p["sub"], 0) if p else None for p in ms.plans]
                    prompt = compact_prompt(self.env.mdp, state, payload.get("score", 0), self.horizon,
                                            keep, [h[-self.k:] if self.k else [] for h in self.hist], self.roles)
                    self.pending[i] = (self.pool.submit(self.choosers[i], prompt, {i: opts},
                                                        deepcopy(state)), held(state.players[i]))
                    info["asked"].append(i)
            # A completed/invalid old plan must not keep interacting while waiting.
            names.append(self.space.stay if needs else inv[ms.step_action(state, i)])
        info["async_pending"] = [p is not None for p in self.pending]
        info["subgoal"] = [p["sub"] if p else None for p in ms.plans]
        return names, None, info
