# Async planner research for Overcleffed

Research date: 2026-10-25. Official documentation and source inspected through
Chrome DevTools MCP. This is a recommendation, not an implemented migration or a
performance result. The research subagent launcher aborted; research ran inline.

## Recommendation

Use a **reactive, tick-driven task executive**, retaining the existing recipe
planner and path planner. `py_trees` is the best-matched off-the-shelf Python
implementation researched. Its action lifecycle directly addresses stale goals:
start once, report RUNNING without blocking, finish on a world-state completion
condition, and terminate when invalidated. [1][2]

A behavior tree is an execution/decision framework, **not an automatic recipe
search planner**. It does not infer recipes, ensure collision-free routes, or
make Clef smarter. Keep `Planner._need`, `Planner._bring`, and benchmark motion
planning; add explicit goal lifecycle semantics at their boundary. This is a
project-specific recommendation based on the existing code, not a library claim.

Ponytail alternative: implement that lifecycle using the existing futures and
small goal records, with no dependency. Prefer this as the first proof if a tree
library would mostly wrap two workers. If we want a reusable inspected executive
for more games, adopt `py_trees` rather than inventing a behavior-tree framework.
Do not add a formal planning engine just to fix cached goals.

## What the primary sources establish

### 1. py_trees: best language/runtime fit

- `initialise()` runs on entry, `update()` reports RUNNING/SUCCESS/FAILURE, and
  `terminate()` runs when work finishes or is interrupted. The documentation
  explicitly says update must not block. [1]
- The official action demo starts work in an external process and polls it on
  subsequent ticks. That is an executable reference for our request/reply
  interaction, although its mocked process is not an HTTP client. [3]
- A Parallel composite ticks all children **sequentially**; it does not create
  threads or processes. Actual slow work must run outside the tick. It returns
  FAILURE if any child fails, so putting two raw chef branches under a Parallel
  without local recovery can unnecessarily stop both. [2]
- A memoryless Sequence rechecks earlier conditions; a memoryful Sequence skips
  earlier checks while resuming a running child. A memoryless Selector permits
  high-priority branches to interrupt lower-priority running work. [2]
- The Timeout implementation uses `time.monotonic()`, invalidates a RUNNING child
  when expired, and returns FAILURE. This only works responsively when updates
  are nonblocking; it is not a preemptive HTTP/thread kill. [4]
- Blackboard clients support namespaced data, access registration, and an activity
  stream useful for debugging. This is not evidence of a transactional multi-agent
  reservation manager; reservations still need application ownership. [5]
- PyPI reports version 2.6.0 and a `pydot>=1.4` dependency. Its `requires_python`
  metadata is null. Our project supports Python 3.10–3.11, so compatibility must be
  tested before pinning a release; do not infer compatibility from missing
  metadata. The cited py_trees pages are development documentation. [6]

### 2. BehaviorTree.CPP: strong reference, poor direct integration fit

Its official `StatefulActionNode` source calls it the preferred asynchronous
Action implementation, particularly for request/reply: `onStart()`, `onRunning()`,
`onHalted()`. [7] This independently supports the proposed lifecycle design.

We are a Python project with existing Python adapters; importing a C++ execution
framework adds integration/build work without resolving recipe logic. Recommendation:
use its design as a reference, not as a new runtime. Its tutorial site failed to
load in this browser session; the claim above was verified against source instead.

### 3. Unified Planning: useful later for actual search, not this repair

Official APIs expose OneshotPlanner, Replanner, and PlanRepairer. Replanner can
update initial fluent values, actions and goals before `resolve()`. AnytimePlanner
returns an iterator of candidate solutions; that alone does **not** make execution
nonblocking. Engine support varies, and optional solve parameters may be ignored
by engines that do not implement them. [8]

Fit: interesting when we need explicit scheduling/resource optimization or novel
recipe search. Not the first choice now: we'd need to encode the domain and
integrate an engine while still writing the async executive and stale-result guards.

## Proposed integration in this repository

### Boundaries to preserve

- `collab_bench/planner.py`: deterministic recipe decomposition and role ownership.
- `collab_bench/clef_agent.py`: Clef grounds a valid goal into benchmark skills.
- `collab_bench/gemma_planner.py`: optional slow plan proposer, never sole owner
  of execution state or safety checks.
- Existing benchmark path planner: navigation and skill execution.
- `overcleffed/async_policy.py`: reusable lessons about one future per chef and
  main-thread state mutation, **not** a drop-in Collab implementation. Its trial
  currently targets standard overcooked_ai with a different adapter.

### Tick contract (proposed)

On every simulation tick, on the simulation thread:

1. Capture one consistent kitchen snapshot for both chefs.
2. Retire completed goals **before** testing whether their old preconditions hold.
3. Invalidate impossible goals and release their resource claims.
4. Accept a returned proposal only if episode ID, goal generation, ownership and
   current preconditions still match. Recompute navigation from current position.
   Do not reject merely because an unrelated world tick changed.
5. Keep a still-valid active skill moving; independently poll each chef's request.
6. Ask Clef only when a new skill choice is needed. Idle a chef with no safe work,
   not the whole simulation. Exactly one environment step consumes joint actions.

Do not run two `env.step()` calls concurrently. Async means independent decision
availability around one authoritative simulation loop, not two owners of the game.

### Minimal goal record (proposed)

Agent, goal ID/generation, action/item/station arguments, status, deadline and
failure reason. Define applicability and completion predicates in code. Separate:

- no useful work (intentional idle),
- waiting for a plan/request,
- running a skill,
- completed,
- invalid/failed/timed out.

The library statuses can represent execution outcomes, but the application still
needs the waiting/failure reasons; `None` must not encode every case.

### Two independently ticked agent branches

Use a small synchronous coordinator for item/station claims, then tick both
agent branches each frame. Local failure handling must allow the other chef to
continue. Each branch checks urgent/current-world conditions, monitors its active
skill, and otherwise submits/polls a Clef request. Wrap long work in deadlines;
perform local recovery instead of failing a whole team Parallel.

For Collab, roles already prevent several conflicts. Initially track only claims
that actually cause handoff conflicts: who is to retrieve a particular counter
item and where it needs to go. Claims need a release condition, not just a lock.
Do not build a general distributed scheduler.

### Boiled egg: world-state termination, not repeated instruction text

| Goal | Complete when | Next eligible work |
|---|---|---|
| Assistant fetches egg | Assistant holds egg | Transfer toward pot-side chef |
| Assistant deposits egg | Egg is at the designated counter, no longer held | Assistant does not retrieve it again without a new need |
| Chef retrieves egg | Chef holds egg | Put egg in pot |
| Chef loads pot | Pot contains required egg, chef no longer holds it | Start pot |
| Chef starts pot | Pot is cooking | Wait/perform other useful work |
| Chef retrieves cooked egg | Chef holds boiled egg | Serve |
| Chef serves | Delivery/reward/terminal condition confirms success | Task complete |

If Gemma times out, preserve only an active goal that remains valid and incomplete.
Otherwise use the deterministic planner or explicit safe waiting/retry. Do not
replay a completed fetch goal or label a planning exception as intentional idle.

### Request cancellation caveat

The existing `ThreadPoolExecutor` can keep requests off the tick thread. Python's
`Future.cancel()` cannot cancel a callable already running; `shutdown` likewise
does not kill running work. [9] Use finite HTTP timeouts, logical invalidation via
generations and bounded outstanding work. Do not start unbounded replacements
while obsolete calls still occupy workers. Hard whole-call deadlines require
appropriate transport/process handling; a tree Timeout alone is not enough.

## How to prove this helps (before a larger migration)

1. Delayed fake planner/Clef: one chef continues while the other waits.
2. Completed fetch + planner exception: no second egg is fetched.
3. Completed counter deposit + missing proposal: no automatic pick-back-up loop.
4. Late response after cancellation/episode reset: never committed.
5. Finished boiled egg held: stale retrieval goal is retired; serving stays possible.
6. Contended counter item: one owner, released on completion/cancellation.
7. Replay the failing snapshot sequence without LLM calls, then run one real
   boiled-egg episode and representative larger recipes.

Compare equal model prompts/options and pacing where possible. Report delivery
success, repeated handoffs, model calls, pending-idle ticks, request latency,
invalidations and timeouts. Async plumbing alone is not a gameplay result; current
5 Hz trials scored 0 soups on both tested standard layouts.

## Decision

**Recommended architecture:** current recipe planner + a reactive goal executive +
async Clef skill selection. **Recommended library if adopting one:** `py_trees`.
**Smallest first step:** explicit goal completion/invalidation and failure state,
using current Python futures; prove the egg loop is gone before adding a framework.
No new dependency or product-code change was made during this research.

## Primary sources

[1] py_trees behavior lifecycle and nonblocking update:
https://py-trees.readthedocs.io/en/devel/behaviours.html

[2] py_trees Sequence/Selector/Parallel semantics:
https://py-trees.readthedocs.io/en/devel/composites.html

[3] Official external-action demo implementation:
https://py-trees.readthedocs.io/en/devel/_modules/py_trees/demos/action.html

[4] Timeout implementation:
https://raw.githubusercontent.com/splintered-reality/py_trees/devel/py_trees/decorators.py

[5] Blackboard design and activity stream:
https://py-trees.readthedocs.io/en/devel/blackboards.html

[6] Published package metadata:
https://pypi.org/pypi/py_trees/json

[7] BehaviorTree.CPP asynchronous/stateful node interface:
https://raw.githubusercontent.com/BehaviorTree/BehaviorTree.CPP/master/include/behaviortree_cpp/action_node.h

[8] Unified Planning operation modes and engine caveats:
https://unified-planning.readthedocs.io/en/latest/operation_modes.html

[9] Python futures cancellation and shutdown semantics:
https://docs.python.org/3/library/concurrent.futures.html
