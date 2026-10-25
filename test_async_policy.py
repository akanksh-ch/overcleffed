"""Run: .venv/bin/python test_async_policy.py"""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from overcleffed.async_policy import AsyncMacroPolicy


class ControlledPool:
    def __init__(self):
        self.calls = []

    def submit(self, *args):
        f = Future()
        self.calls.append(f)
        return f


class Plans:
    def __init__(self):
        self.plans = [None, {"sub": "get_onion"}]
        self.steps = []

    def needs_subgoal(self, state, i):
        return self.plans[i] is None

    def options_for(self, state, i):
        return {"get_onion": 1, "get_dish": 1}

    def assign(self, i, sub, state):
        self.plans[i] = {"sub": sub}

    def step_action(self, state, i):
        self.steps.append(i)
        return (1, 0)


p = AsyncMacroPolicy([None, None])
p.pool.shutdown()
p.pool = ControlledPool()
p.ms = Plans()
p.env = SimpleNamespace(mdp=None)
p.hist, p.horizon = [[], []], 20
s = SimpleNamespace(players=[SimpleNamespace(held_object=None), SimpleNamespace(held_object=None)])
with patch("overcleffed.async_policy.compact_prompt", return_value="snapshot"):
    for _ in range(3):
        actions, _, info = p.act({}, s)
        assert actions == ["stay", "right"]  # chef 1 progresses despite pending chef 0
    assert len(p.pool.calls) == 1  # never queue duplicate requests
    p.pool.calls[0].set_result(({0: "get_onion"}, {}, {"called": True}))
    actions, _, _ = p.act({}, s)
    assert actions == ["right", "right"]
    p.ms.plans[0] = None
    p.act({}, s)
    s.players[0].held_object = SimpleNamespace(name="dish")
    p.pool.calls[1].set_result(({0: "get_onion"}, {}, {"called": True}))
    actions, _, info = p.act({}, s)
    assert info["async_discarded"] == 1 and actions[0] == "stay"
    assert len(p.pool.calls) == 3  # stale choice discarded and fresh request submitted
print("PASS: independent progress, one request per chef, stale response rejection")
