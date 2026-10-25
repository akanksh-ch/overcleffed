# Mini Metro feasibility spike

## Decision

**Simulator feasibility passed; model integration not yet attempted or approved.** Use the upstream structured environment for the next bounded experiment, retaining the kitchen demo as fallback. No checkpoint loading, training, paid inference, or remote service changes were performed.

## Isolation and provenance

- Branch: `mini-metro-spike`.
- Upstream: https://github.com/yanfengliu/python_mini_metro
- Exact commit: `ef1be2dfdc6e1dadafa9bf2c41d1ce5f081adc4d` (MIT).
- External checkout: `/tmp/overcleffed-mini-metro-upstream`.
- External environment: `/tmp/overcleffed-metro-venv`; CPython 3.13.16, Linux x86_64. Installed using uv after inspecting upstream README and requirements. Repository `.venv`, dependency declarations, and lockfile untouched.
- Installed versions: gymnasium 1.3.0, numpy 2.5.4, pygame-ce 2.5.8 (SDL 2.32.10), shapely 2.2.0, shortuuid 1.0.13, pillow 13.0.0, pytest 9.1.1; cloudpickle 3.1.2, farama-notifications 0.0.6, iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0, pygments 2.21.0, typing-extensions 4.16.0.
- Used minimal `requirements.txt`, not upstream locked resolution; versions above record this particular execution. No Node setup or RL dependencies installed.

## Executed checks

With dummy SDL video/audio drivers, a 90-second bounded pytest invocation ran:

```
python -m pytest -q test/test_env.py test/test_headless_render.py \
  test/test_render_purity.py test/test_passenger_flow.py test/test_simulation_context.py
```

Result: **49 passed, 10 subtests passed in 1.43 s**. This is a focused subset, not the full upstream suite.

Reset/step and route probe used seed **42**, Classic defaults, unmodified game configuration. Fixed route means `create_path(stations=[0,1,2])`, then `assign_locomotive(path_index=0)`, both with `dt_ms=0`, followed only by noop steps. This is a transport smoke test, **not continuous reactive policy control**.

| Control | Step duration | Steps | Simulated time | Deliveries / return | End | Waiting | Wall seconds |
|---|---:|---:|---:|---:|---|---:|---:|
| noop | 16 ms | 2500 | 40.000 s | 0 / 0 | game over | 9 | 0.119 |
| fixed route | 16 ms | 6916 | 110.656 s | 17 / 17 | game over | 9 | 0.805 |

The fixed route unlocked a fourth station but never connected it. This supplies a concrete reason for further decisions during an episode.

A preliminary **100 ms per step** probe also executed: noop ended at 40 s with 0 deliveries and 3 waiting; fixed route reached the 180 s cap with 6 deliveries, 0 waiting, and 1 onboard. Repeating that fixed-route run reproduced these metrics and the final rendered RGB SHA256 `246cf6b11be8b25633483e981b681a5bea1d52bf4431911b51737cc147c8e2d4`. Headless 1920×1080 PNGs were generated at `/tmp/metro-noop.png` and `/tmp/metro-fixed_route.png` (these depict the preliminary 100 ms runs).

**Important timing limitation:** passenger spawn cadence uses step counts while travel/wait timing uses milliseconds. Simply increasing `dt_ms` changes demand relative to movement; it is not a neutral speedup. Use fixed 16 ms physics ticks for any comparison and advance multiple physics ticks between decisions. Even zero-duration action steps increment simulation steps, so action scheduling must be matched across controls. The 100 ms results are not comparable to the 16 ms results.

## Semantics and limitations

- Default reward is newly delivered passengers; summed return matched lifetime deliveries in these probes. Legacy line-credit reward is a separate selectable objective.
- Default game over is the second station passenger exceeding the 40-second waiting limit, not merely reaching station capacity. Waiting counts alone do not fully describe risk.
- Station unlocks depend on deliveries (first at 10), not just elapsed time. Avoid claiming this precisely reproduces commercial Mini Metro demand dynamics.
- Structured observations expose station geometry/shapes, queues, passenger destinations, routes, trains, fleet, deliveries, and time. Atomic route operations are assisted control, not human-equivalent pixel play.
- Rendering works through `GameRenderer.draw(surface, mediator, alpha=1.0)`; the structured environment itself is not a ready-made video dashboard.
- Seed determinism excludes opaque session entity IDs. Only the preliminary fixed-route run was independently repeated here.
- Wall times above measure fast local execution, not inference latency or real-time interactive performance. No model success, model errors, or stale-response behavior has been measured.

## Smallest proposed next experiment — requires user approval

Existing project code exposes Clef `/v1/systemone` choice questions. A distinct Jev interface was not found in the inspected project Python code; do not assume branding or API equivalence without clarification.

1. Keep upstream external and pinned. Add only a small local runner/adapter plus focused tests if approved.
2. Run fixed 16 ms physics ticks, request decisions at a fixed simulated cadence, and keep simulation advancing while inference is pending. Bound wall duration and HTTP timeout. At most one request in flight; log snapshot age and revalidate actions against current state before applying. Reject stale/invalid responses explicitly rather than silently substituting strategy.
3. Enumerate a bounded, explicitly documented menu: noop, create a two-station route, extend either endpoint to a station not already on that route, assign/unassign a locomotive, and attach/detach a carriage. Include resource/legal checks but no destination-aware strategic ranking or hidden route solver. Stable IDs must map safely to current upstream indices. The restricted operation set itself is a task limitation.
4. Give the model compact structured state and descriptions of the legal menu. Keep the heuristic separate: connect unserved stations to a nearby existing endpoint and allocate spare trains using an explicit queue-pressure rule. Do not attribute this heuristic to the adapter or model.
5. Compare heuristic and model on the same small predeclared seed set and identical physics, action cadence, episode caps, and latency/deadline treatment. Decisions must remain active throughout; initial-network construction alone does not qualify. Log deliveries, waiting/overdue risk, game-over time, simulated/wall time, decision count, inference latency, errors, and stale/rejected actions. Report timeout behavior and fallback contributions separately.

## Approved continuous-control implementation

The user approved the existing **Clef** `/v1/systemone` endpoint, with no service/model-setting changes, seed 42, heuristic versus Clef, and per-run caps of 120 simulated seconds / 150 wall seconds. Physics ticks are 16 ms, with one decision opportunity every 5 simulated seconds, at most 24 requests, one in flight, and a 3-second acceptance deadline. HTTP uses a finite 2.5-second socket timeout. This is a Clef experiment, not a separately verified Jev API integration.

`mini_metro_spike.py` contains the small standalone adapter, baseline, asynchronous request boundary, paced runner, and GIF export. `test_mini_metro_spike.py` exercises the real external simulator with delayed/error policies rather than real inference. It does not import kitchen code. Only detached state and the legal menu enter the inference worker; snapshot/episode checks and renewed current-state legal enumeration happen on the main thread. No policy fallback is used. Deadline expiration leaves the request in flight until its worker finishes; a second request is never started in the meantime. Actions replace a noop on the next ordinary physics tick, rather than adding zero-duration steps that would change spawn cadence.

The restricted menu includes noop, every unordered two-station route, extension of either non-loop endpoint to any station not already on that route, locomotive assignment/unassignment, and carriage attachment/detachment. Fleet predicates reuse upstream legality queries. Route enumeration is Classic-only, with no destination/distance ranking or top-k selection. It deliberately omits arbitrary reroutes, route removal, loops, line purchasing, and upgrades. The heuristic is separate: start with a nearby unlike-shape pair, assign unserved lines a train, connect unserved stations at the nearest existing endpoint, and add capacity above an explicit queue-pressure threshold.

### Reproduction

Assuming the external checkout and environment recorded above:

```sh
/tmp/overcleffed-metro-venv/bin/python -m unittest -v test_mini_metro_spike
# Existing localhost:8080 tunnel must already lead to the approved Clef service.
timeout 310 /tmp/overcleffed-metro-venv/bin/python mini_metro_spike.py \\
  --upstream /tmp/overcleffed-mini-metro-upstream \\
  --output runs/mini_metro_seed42
```

The runner verifies upstream commit and tracked cleanliness, refuses an existing output directory, and saves manifest, per-policy summaries, decision events (including menu/state/probabilities/errors), sampled congestion telemetry, and a labeled GIF/final PNG. Artifacts are privileged structured-state visualizations, not pixel-model observations. Physics runs at real-time pace, catching up with unchanged 16 ms ticks after local overhead; final GIF encoding is outside the episode cap and separately timed. A request pending at episode end is logged and discarded, never joined or applied to another episode.

Visual smoke inspection caught an important renderer usage detail: `GameRenderer.draw` does not clear the surface. The recorder now fills upstream `screen_color` before every frame, matching the upstream human loop; a regression test covers clearing. Preliminary `/tmp/metro-*.png` images and hashes above came from the original uncleared probe surface and should not be used as polished visual evidence.

No reason to begin training or evaluate untrusted PPO checkpoints for this hackathon spike. Real comparison results follow once measured.

## Measured results: seed 42, one run per policy

Run 2026-10-25 via the reproduction command above (artifacts in ignored `runs/mini_metro_seed42/`: manifest, per-policy summary/events/telemetry, GIF, final PNG). Caps, cadence and deadline as approved; no service or model-setting changes. Clef is the existing `clef-flash` model on the existing localhost:8080 tunnel.

| Policy | Deliveries | Sim end | Wall | Calls (accepted) | Waiting at end | Peak waiting | Peak oldest wait | Game over |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Heuristic | 30 | 120 s (cap) | 120.07 s | 24 (24) | 1 | 4 | 18.4 s | no |
| Clef | 23 | 120 s (cap) | 120.07 s | 24 (24) | 5 | 6 | 31.1 s | no |

No overdue passengers, game-over, late, deadline-expired, error, or upstream-rejected actions in either run. Clef latency: median 0.86 s, max 0.98 s; accepted actions applied up to 1.008 simulated seconds after their snapshot (heuristic: 16 ms). Both runs ended at 4 stations with one line.

Decision sequences (24 opportunities each):
- Heuristic: create S0-S2, assign, extend to S1, noop x7, extend to S3, noop x4, assign, noop x7, assign.
- Clef: noop (first opportunity wasted), create S0-S1, assign twice, extend to S2, attach/attach/detach/attach carriage, noops, extend to S3 at t~60 s, unassign, attach, noops, unassign, assign.

## Interpretation and limits

- Single seed, single run each, one model. This is a smoke comparison, not a benchmark; 30 vs 23 deliveries is not evidence of a general ranking. No statistical claim.
- Clef was usable on the continuous loop: every call answered inside the 3 s deadline, none stale-rejected, no fallback used. It kept managing (extend, fleet changes) through the episode, not only at start.
- Clef trailed the heuristic here. Visible causes in the log: a wasted first opportunity, a later connection of the fourth station (S3 at ~60 s vs ~50 s), and fleet churn (carriage detach, locomotive unassign/assign). Reasons are inferred from logs, not tested.
- Timing treatment is not equal in one respect: the heuristic answers in microseconds, Clef in ~0.86 s, so Clef acts on a staler snapshot. This is the realistic latency cost, but means the gap mixes decision quality and latency. A latency-matched heuristic (add 0.86 s delay) was not run.
- Both runs ended on the 120 s cap, not game over; congestion pressure (peak oldest wait 18-31 s of the 40 s limit) never reached failure. A longer or denser episode could separate the policies more, but was outside the approved scope.
- Assisted control only: structured state, atomic route/fleet operations, privileged-view GIF. Not pixel play, not human-equivalent. Clef only; no Jev interface tested.
- Non-determinism: Clef responses and wall scheduling were not repeated; only upstream mechanics seed 42 is fixed.
- Visual artifact checked by eye: Clef final frame shows a single orange line with carriages and one off-view/unconnected station state is not verified from the image; trust telemetry/events JSON over the GIF for metrics.

Next options (not run, need approval): 3-5 more seeds each, latency-matched heuristic control, a tighter demand/longer cap so congestion actually bites, or stop and keep the kitchen demo as primary.

## Seed sweep at default demand (spawn interval 900 steps, 120 s cap)

- Heuristic: seeds 1-8, 0 game-overs, 28.0 deliveries mean (14.0/min), mean waiting 1.95, peak oldest wait 21.2 s.
- Clef (choice menu): seeds 1-5 complete, 2 game-overs (seeds 3 and 4), 22.0 deliveries mean (12.0/min), mean waiting 3.47, peak oldest wait 40.4 s; latency median 0.89 s, max 1.81 s, no errors.
- Paired on seeds 1-5: Clef never beat the heuristic on survival (0 wins, 3 ties, 2 losses; mean -14.4 s) or deliveries (0 wins, 2 ties, 3 losses; mean -6.8).
- Clef seeds 6-8 are incomplete (sweep killed for time); not rerun, excluded. Discarded: first Clef pass with 6 parallel runs had 21 `TimeoutError` on the 2.5 s socket timeout (shared endpoint load), moved to `runs/mini_metro_sweep/spawn900/_clef_parallel6_timeouts/`.
- Stress levels beyond default spawn (denser demand) were not run. Game-overs came from Clef connecting stations late and churning fleet, per event logs; not isolated experimentally. n=5, one run each, no significance claim.
