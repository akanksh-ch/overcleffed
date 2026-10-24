# Visualising Clef playing Overcooked

## What overcooked_ai ships (v1.1.0, checked in `.venv/.../overcooked_ai_py/`)
- `visualization/state_visualizer.py::StateVisualizer`: pygame sprite renderer. `render_state(state, grid, hud_data, action_probs)` returns a `pygame.Surface`; `display_rendered_state(..., img_path=, window_display=)` saves PNG or opens a window; `display_rendered_trajectory(trajectories, ...)` dumps per-step PNGs. `action_probs` (per chef, 6 values in `Action.ALL_ACTIONS` order) draws arrows/icons sized by sqrt(prob) next to the chef: a ready-made "how it thinks" overlay.
- Runs headless with `SDL_VIDEODRIVER=dummy`. pygame 2.6.1 wheels install via uv on Python 3.11 with no system packages. Live windows were not tested here (Wayland goes through SDL's XWayland/Wayland backend; set `SDL_VIDEODRIVER=wayland` or `x11` if the window fails).
- `overcooked_demo` (Flask + socket.io, in the GitHub repo, not the pip package; unverified here, no network fetch done): browser human-vs-AI play, server-side game loop at fixed tick rate, pluggable agents (`server/game.py` `OvercookedGame`, `Agent.action(state)`). Needs the repo cloned, old pinned Flask/socketio, and docker is the supported route. The pip package has no replay tool beyond `display_rendered_trajectory`.
- Log has no state, so replay re-simulates: env is deterministic given layout and actions (`env.step`).

## Options
| | Effort | Fun | Benchmark-safe | Notes |
|---|---|---|---|---|
| (a) live pygame window | 0.5d | high | yes if render off the tick thread | window cost eats tick budget; use a render thread reading a queue |
| (b) browser dashboard (FastAPI+ws, canvas) | 2-3d | highest | yes | reimplement rendering in JS or stream PNGs |
| (c) GIF/MP4 per episode from JSONL | done (prototype) | good, shareable | perfect (offline, zero tick impact) | |
| (d) overcooked_demo with Clef agent, human plays with/against | 2-4d | highest for human-vs-Clef stress study | partial (different loop than run.py; scoring is the same env/`reward`) | Clef agent = wrapper calling `Policy.act` in a thread, returning `stay` when late (same rule as run.py) |

## Recommendation
1. Now: (c) offline replay, from the log. Zero risk to timing. Built.
2. Next (~0.5d): a live pygame mode in run.py, using `render_state(action_probs=...)` fed from the same `probs`, rendered on a separate thread.
3. For the human-vs-Clef comparison (~3d): (d). Write a small pygame human-play loop with the same env/tick rules instead of porting overcooked_demo's Flask stack, unless a browser UI is a hard requirement. Reuse `policy.act` and the same log schema, so `replay.py` works on human runs too.

## Built: `viz/`
- `viz/replay.py LOG.jsonl -o out.gif [--ep N --budget 0.2 --every 2 --fps 10 --max N --layout L]`: game frame (StateVisualizer, with HUD), per-chef 6-way probability bars (chosen action coloured, entropy in bits), timeline (score, latency bars, red when `late` or over budget, budget line, cursor). Tolerates missing probs (mock/greedy logs), blank/bad lines, probs as dict or list. Layout and horizon come from `LOG.summary.json` `args`.
- `viz/make_fake_log.py IN.jsonl OUT.jsonl`: adds synthetic probs and latency to a real action log.
- `viz/demo.gif`: 200 frames from `runs/greedy.jsonl` plus fake probs.
- Run: `SDL_VIDEODRIVER=dummy uv run python viz/replay.py runs/tick.jsonl -o viz/x.gif`.
- Deps added via `uv add`: pillow, imageio (imageio unused so far; drop it if MP4 stays out of scope).

## Issues / gaps
- `history_k` window is not shown (not in log). Log `history` or the raw Clef response per step to add it.
- Bars show Clef's logged distribution only when `probs` is logged; chosen action under temperature sampling may differ from argmax, which the highlight handles (it uses `actions`).
- Late ticks log the substituted `stay`; bars are absent when probs is null.
- MP4 not done (needs ffmpeg plugin: `uv add imageio-ffmpeg`).
- Replay assumes the env start state is deterministic; true for fixed layouts.
