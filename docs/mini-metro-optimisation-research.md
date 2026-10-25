# Mini Metro: how to improve Clef/Jev, and better alternatives

Research performed 2026-10-25 with Chrome DevTools against vendor docs, model cards and papers. Recommendations are hypotheses for the coding pane to test; nothing here is a measured Mini Metro result.

## What decision models are, and are not

Jev (TypeSafe), Clef / Clef-flash (Cloudflare) and GPT-6 Luna (OpenAI Decisions API) all answer typed questions about a state with probabilities: `choice`, `noul`/predicate and `score`. They do not generate text and do not search. Clef runs one prefill pass and scores every option in parallel ([blog](https://blog.cloudflare.com/clef-decision-models)).

Vendor benchmark on the [Clef model card](https://huggingface.co/Cloudflare/clef) shows where this matters for us:

| Benchmark | Clef | Clef-flash | Jev |
|---|---:|---:|---:|
| ChessBench (accuracy) | 24.7 | 23.0 | 17.2 |
| Home appliance simulator (case exact) | 83.0 | 97.7 | 52.3 |
| BFCL tool calling (case exact) | 98.5 | 98.8 | 95.8 |
| Median latency (ms) | 209 | 39 | 524 |

Decision models are very good at picking the right action for an obvious, observable situation (tools, appliances, routing). They are weak at multi-step lookahead (chess). Mini Metro route design is a lookahead problem: an action pays off tens of seconds later. Asking Clef "which route edit is best?" uses it for the thing it is worst at.

Our measured Clef latency (~0.8 s) comes from Clef-flash Q4 GGUF on an A16 through llama.cpp. The vendor median for Clef-flash is 39 ms on Cloudflare hardware. Part of our gap is hardware, not the model.

## How to use Clef better (ranked by expected payoff per hour)

1. **Simulator lookahead, Clef only as a filter.** Upstream `python_mini_metro` already snapshots and restores full state including RNG (`src/save_load.py`, `src/path_replacement_snapshot.py`). For each legal candidate, clone the state, roll forward N simulated seconds with a cheap default policy, score deliveries and overdue risk, and pick the best. This is model-predictive control. It needs no model at all and is the strongest baseline to beat. Clef then adds value by pruning the candidate list before the expensive rollouts, which is the LLM-MCTS pattern (Zhao, Lee, Hsu, NeurIPS 2023, [arXiv:2305.14078](https://arxiv.org/abs/2305.14078)): the model supplies a prior and search verifies it.
2. **Ask Clef observable questions, let code act.** Replace the single strategic `choice` with questions it is good at: `score` urgency per station ("how close is this station to overflowing?"), `noul` per candidate ("does this action connect the most overloaded station?"). One request can carry up to 64 questions. Code combines the answers with fixed rules. This matches what the project already learned in the kitchen: "code plans, Clef acts".
3. **Score every candidate in one call.** One `score` question per candidate action, all in one request, gives a calibrated ranking instead of a single argmax. Use the probabilities as rollout priors (point 1) or a confidence gate: below a threshold, keep the current network.
4. **Cut staleness.** Each decision is applied to a world that has moved ~0.8 s. Either revalidate against the expected state (already done) or move to Clef-flash on Workers AI (vendor 39 ms median, untested by us, needs a Cloudflare account). Also remove fields from the state that do not affect the choice; Clef truncates long text state.
5. **Fine-tune (later, not this hackathon).** Cloudflare announced an RL fine-tuning service for Clef, starting as an FDE engagement. Rollout-scored decisions from point 1 would be the training labels. Not self-serve today.

## Alternatives to a decision model

- **Rollout / MCTS planner on the simulator.** Best fit for the problem: deterministic simulator with save/restore, small legal action set, slow dynamics relative to decision cadence. Expected to beat both the current heuristic and Clef without any model.
- **Learned transit planners.** Holliday et al. (GNN + evolutionary search, public GPL-3.0 code and weights) and Poudel & Li (graph-attention PPO). They solve static route design for a demand matrix, not live play. Use offline to propose a network shape, if at all. See `mini-metro-literature-followup.md`.
- **Trained RL policy.** Upstream recommends recurrent PPO and documents a long training and evaluation protocol. Too slow for this hackathon.
- **GPT-6 Luna (OpenAI Decisions API).** Same question types, JSON input, up to 400 questions per request. Needs an API key; ChatGPT sign-in does not work. Same planning limitation as Clef.

## Recommended next experiment

On the same seeds and pressure levels the coding pane is already running, compare four controllers:

1. current heuristic,
2. current Clef `choice`,
3. rollout planner (no model),
4. rollout planner with Clef pruning or per-station urgency scores.

If 3 beats 2 by a wide margin, the demo story is "decision model plus a look-ahead check", not "decision model alone". If 4 beats 3 at equal compute, Clef has earned its place.
