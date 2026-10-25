# Async Clef trial checkpoint

## Protocol

One deterministic episode per layout, 400 simulation steps. Baseline: existing
macro-action policy with roles, synchronous inference. Async: independent per-chef
requests, separate HTTP sessions, 5 Hz simulation; a chef stays while awaiting a
new goal and its partner may continue. At most one outstanding request per chef.
Returned goals are revalidated and paths recomputed against current state.

These are not throughput-equivalent runs: baseline pauses simulated time during
inference; async consumes simulation steps while waiting. Async also changes the
request from a joint question to a per-chef question. This comparison does not
isolate which change causes the score difference.

| Layout | Synchronous soups | Async soups |
|---|---:|---:|
| asymmetric_advantages | 19 | not run |
| coordination_ring | 7 | 0 |
| forced_coordination | 0 | 0 |
| counter_circuit | 0 | not run |

All listed episodes completed 400 steps with no runner failures. Async logs
contain no request errors or discarded responses. Forced coordination had
pending requests on 392/400 ticks; coordination ring on 243/400 ticks. At least
one chef took a non-stay action on 237 and 23 ticks respectively. Async plumbing
works, but the trial does not demonstrate a gameplay improvement.

Checks: `.venv/bin/python test_async_policy.py` verifies independent progress,
one pending request per chef, and stale response rejection.

## Gemma

The initial six-recipe-per-model queue was stopped. A new hosted 26B boiled-egg
trial omitted the explicit `max_tokens` field while retaining HTTP timeouts and
an external 180-second process deadline. It exited 124 (deadline reached) without
writing a completed episode result. Omitting the parameter does not remove the
provider's default output or context limits. No success improvement established.

## Artifacts

- Baseline: `runs/larger_20261025/`
- Async: `runs/async_20261025/`
- Remote no-cap trial: `/opt/collab/src/resume_no_cap/`
- Server-rendered reel destination: `runs/video/server_render/demo_reel_server.mp4`

Next useful experiment: keep the per-chef request protocol identical and sweep
simulation pacing or compare lockstep vs parallel calls separately. Diagnose the
existing counter-handoff/blocked-path failures before changing models or games.
