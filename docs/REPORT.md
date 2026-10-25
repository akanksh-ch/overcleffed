# Overcleffed: a decision model cooks in a two-chef kitchen

Cloudflare's Clef-flash is a 9B *decision model*: it does not write text, it returns a
probability for each allowed option in one fast prefill pass. We tested whether it can
run two cooperating chefs in Collab-Overcooked (EMNLP 2025), a benchmark of 30 recipes in
6 difficulty levels where the chef and the assistant stand on opposite sides of a wall and
must pass items across a shared counter.

Everything runs self-hosted on a Vultr GPU instance (NVIDIA A16) with llama.cpp.

## Architecture: "code plans, Clef acts"

Feudal (manager/worker) hierarchy:

1. **Strategic layer: planner** (`collab_bench/planner.py`). Plain code. Reads the recipe
   tree, knows which stations each agent can reach, inserts the counter handoffs the recipe
   text never mentions, and recomputes one goal per agent from the game state every tick.
2. **Operational layer: Clef-flash** (`collab_bench/clef_agent.py`). Gets a ~400-character
   JSON state (its goal, holdings, utensils, counter items) and one `choice` question whose
   options are the benchmark's valid skills, each with a plain-language description. Idle
   (`wait`) is masked while there is work.
3. **Motion layer:** the benchmark's own path planner walks the agent to the chosen target.

This follows Cloudflare's guidance for decision models: keep the state small, make options
literal, and do multi-step logic in code.

## Results (30 tasks, 120-tick limit, 1 run per task)

| Level | Clef alone | Planner + random skill | **Planner + Clef** | Ticks used |
|---|---|---|---|---|
| 1 | 4/5 | 0/5 | **5/5** | 16-17 |
| 2 | 2/5 | 0/5 | **5/5** | 24 |
| 3 | 0/5 | 0/5 | **5/5** | 43 |
| 4 | 0/5 | 0/5 | **5/5** | 40 |
| 5 | 0/5 | 0/5 | **5/5** | 52 |
| 6 | 0/5 | 0/5 | **5/5** | 59-65 |
| Total | 6/30 | 0/30 | **30/30** | |

- Clef latency per decision fell from about 0.45 s (1,100-token LLM-style prompt) to
  0.07-0.24 s (compact JSON state).
- **Clef alone** fails at handoffs: after cutting a potato it picks `wait` (0.70) over taking
  the slices off the board (0.19), the state never changes, and it repeats until time runs out.
- **The ablation** (same planner, random skill choice) solves nothing, so Clef's choices matter.

## Honest caveats

- The planner does the reasoning. 68% of decisions (1,154 of 1,701) had only one valid option
  and did not reach Clef; the other 547 averaged 3.4 options and Clef chose at 0.98 confidence.
  Clef's job here is fast, reliable grounding of a goal into the right skill.
- Not comparable to the paper's LLM table (GPT-4o 94% at level 1, 4% at level 6): those
  agents plan for themselves and coordinate through chat, with no hand-written planner.
- 1 run per task (Clef with argmax is deterministic); 120-tick limit, looser than the paper's.

## Replacing the script planner with Gemma 4 12B

We tried an LLM (Gemma 4 12B, Q4, same GPU) as the strategic layer, with typed output:
a JSON schema that only allows stations the agent can reach, plus code that checks each goal's
preconditions against the state and sends rejected goals back with the reason. See the
results file for the latest numbers; in early runs Gemma failed to start appliances, used
ingredients the recipe does not need, and was slow on the A16 (about 18 tokens/s, 3-10 s per
call).

## Files

- `collab_bench/planner.py`, `clef_agent.py`, `gemma_planner.py`, `run_clef.py`: the agents and runner.
- `collab_bench/render_gif.py`, `side_by_side.py`: GIFs with the planner goal and Clef's probabilities.
- `runs/gifs/`: `hier_*` (planner + Clef), `flat_*` (Clef alone), `compare_*` (both).
- `docs/decisions.md`: decision log.
