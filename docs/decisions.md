# Decision log

## 2026-10-25: Vultr instance spec
Chose Cloud GPU 1x A16 (16 GB VRAM, $0.471/hr) over A40/L40S (48 GB, from $1.712/hr). Clef-flash Q4 GGUF is 4.9 GB, so 16 GB fits it with room for a second model. $100 buys about 212 hours.

## 2026-10-25: Serving Clef
Used the prebuilt llama.cpp b11869 CUDA 13.4 release binary instead of compiling. Server listens on 127.0.0.1:8080 only; access is over SSH tunnel. Needs `-ub 4096 -b 4096` or `/v1/systemone` rejects prompts over 512 tokens.

## 2026-10-25: overcooked_ai results (cramped_room, 400 steps)
Raw actions: 0 soups. Macro actions: 1 soup. Macro + `--roles`: 5 soups in 190 steps. Greedy scripted baseline: 9 soups. Random subgoals with the planner executing them: about 11 soups per episode. Conclusion: most of the score comes from the planner and the human-written roles, not from Clef. Latency 2-4 s per step on the A16 rules out real-time play with one call per step.

## 2026-10-25: Move to a harder benchmark (Collab-Overcooked, EMNLP 2025)
Options considered: real-time overcooked_ai, a live pygame window, human + Clef play, or a harder benchmark. Chose Collab-Overcooked: 30 recipes in 6 difficulty levels, published success rates for 13 LLMs, and roles already built in (chef works pot/oven and serves; assistant works chopping board/blender and fetches).

## 2026-10-25: How Clef plugs into Collab-Overcooked
Subclassed the benchmark's `LLMAgents` and replaced only `generate_ml_action`. Each chef calls Clef separately when it needs a new skill. Options are the benchmark's own skills (`pickup(x, place)`, `cook(pot0)`, ...), filtered by the benchmark's validator. The benchmark's path planner executes the chosen skill.

Deviations from the paper's protocol, which must be stated when comparing:
- No natural-language messages between agents (Clef returns choices, not text). To compensate, both agents see the recipe; in the paper only the chef knows it.
- Invalid skills are removed before Clef chooses. In the paper, the LLM proposes freely and gets one self-correction attempt.
- One run per task (Clef argmax is deterministic) instead of 10.

## 2026-10-25: Adapter bug found and fixed
Clef chopped a bell pepper that the recipe did not need. The candidate list only had recipe items, so `bell_pepper_slices` could never be picked up again and both chefs waited for 110 steps. Fix: candidates now include every item currently in the kitchen. First 11-task run discarded.

## 2026-10-25: Flat Clef results on Collab-Overcooked (30 tasks, 120 steps, 1 run each)
Level 1: 4/5. Level 2: 2/5. Levels 3-6: 0/20. Failure mode: Clef picks `wait(1)` (about 0.70) over the needed handoff step, the state does not change, and it repeats until time runs out. The recipe text never mentions counter handoffs, so Clef cannot infer them. The prompt was about 1,100 tokens of benchmark LLM text.

## 2026-10-25: Hierarchical redesign (feudal manager/worker)
Followed Clef guidance (keep state small, use literal option text, do multi-step logic in code) and feudal RL (a slow manager sets subgoals; workers act; each subgoal has a termination check).
- Strategic layer: `collab_bench/planner.py`, plain code, not Clef. Walks the recipe tree, knows which side of the wall each station is on, inserts counter handoffs, and recomputes one goal per agent from the game state every tick.
- Operational layer: Clef picks a skill toward its goal. Compact JSON state of about 400 characters. Each option has a plain-language description. `wait(1)` is masked while there is work. An agent with no goal idles without calling Clef, which stopped both agents chasing the same item.
Result: 30/30 tasks succeed, including all level 6. Mean Clef latency 0.07-0.24 s, down from about 0.45 s.

## 2026-10-25: Ablation, and where the credit goes
Same planner with a uniform random worker: 0/30, including level 1. Clef's choices matter. But 68% of decisions (1,154 of 1,701) had only one valid option and skipped Clef. The remaining 547 averaged 3.4 options at 0.98 confidence. The planner does the reasoning; Clef grounds goal text into the correct skill. Present it as "code plans, Clef acts", not as Clef solving the benchmark. Not comparable to the paper's LLM numbers, which have no hand-written planner.

## 2026-10-25: Collab goal lifecycle correction
Implemented the bounded repair in `collab_bench/gemma_planner.py`, without new dependencies or changes to Clef/model settings. Every planner read retires observed completed goals before checking their old preconditions. Fetch/pickup/retrieval completes when held; deposit when released and the item is located in the world; loading/start/wait/fill use utensil or holding effects. Serving is not inferred complete from empty hands (no delivery evidence at this boundary).

Failed proposals retain only valid unfinished typed goals. Other agents use the existing deterministic `Planner`, with per-agent `sources` (`model`, `intentional_idle`, `retained`, `fallback`), `retired` reasons, and non-null call `err` on final failure. Fallback with no work remains explicitly fallback, not intentional model idle. Validate response envelopes and action/item/station enums locally as well as through provider schema constraints. Shared callers are both Clef agents and runner trajectory logging; neither requires an interface change.

Proof: `test_collab_goal_lifecycle.py` initially failed (5 failures, 7 errors across 7 tests/subtests); all 7 now pass locally and on the Collab server. Cases cover completed fetch/deposit/retrieval, valid unfinished work, failure versus idle, malformed typed responses, and completion-before-precondition ordering. Existing `test_async_policy.py` passes; compileall and `git diff --check` pass.

One real local-Gemma/Clef boiled-egg smoke, external timeout 180 seconds, new output `runs/lifecycle_smoke_20261025/`: success, 26 ticks, reward 20, 97.948 seconds, exit 0. Ten planner calls: seven model goals, twelve intentional-idle agent decisions, one deterministic fallback, one failed call, five completed-goal retirements. Credit is mixed model + deterministic recovery, not pure LLM success. Remote baseline planner hash matched HEAD before replacement; only planner and test synced. Results fetched locally (runs are ignored by git). A final malformed-envelope guard was tightened after the smoke and rechecked with deterministic tests; no second real episode was run.

Deferred: Collab async proposer/execution, generations/late-result handling, resource reservations, and executive/framework adoption. Calls remain synchronous, with existing finite HTTP timeouts; existing standard-overcooked async tests do not prove Collab async. No broad model sweep or service restart.

## 2026-10-25: Async planner research (recommendation, not implementation)
Inspected official py_trees, BehaviorTree.CPP, Unified Planning and Python futures docs/source using Chrome DevTools MCP. Best-fit off-the-shelf Python executive: py_trees, with nonblocking RUNNING/SUCCESS/FAILURE actions and explicit invalidation/termination. A behavior tree is an executive, not an automatic recipe planner. Retain the current recipe decomposition and motion planners; first prove goal completion/invalidation fixes the stale-goal egg loop using existing futures. Framework adoption is not yet approved or implemented. Parallel composites do not create threads; futures cannot kill already-running HTTP requests. See `docs/async-planner-research.md` for citations, integration boundaries and acceptance checks.
