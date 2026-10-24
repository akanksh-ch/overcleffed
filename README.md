# overcleffed

Cloudflare Clef-flash (decision model) plays [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai).
One `/v1/systemone` call per step, two `choice` questions (`chef_0_action`, `chef_1_action`) over the 6 raw actions.

## Run
```
uv sync
uv run python run.py --mock --out runs/mock               # no server, random policy
uv run python run.py --backend greedy --out runs/greedy   # overcooked_ai GreedyHumanModel baseline
# real Clef: start llama-server on :8080 first
uv run python run.py --layout cramped_room --horizon 400 --history-k 5 --decode argmax --out runs/clef
uv run python run.py --decode sample --temperature 0.7 --tick-hz 2 --episodes 3 --out runs/clef_rt
```
Outputs: `<out>.jsonl` (per step: actions, probs, reward, score, latency, late) and `<out>.summary.json`
(soups, total reward, mean/p95 latency). `--tick-hz N` = real-time: a late Clef reply makes both chefs `stay`.
The first raw Clef response is printed to stderr (parser is tolerant; see `parse_choice`).

## Layout
- `overcleffed/env.py` env, state serialization, history
- `overcleffed/policy.py` `RawActions` (swap for macro-actions), Clef/mock/greedy policies
- `run.py` CLI runner
