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

**Next recommendation:** approve this narrow continuous-control comparison only after identifying which Jev/Clef endpoint to use and agreeing on the concrete inference/time budget. No reason to begin training or evaluate untrusted PPO checkpoints for this hackathon spike.
