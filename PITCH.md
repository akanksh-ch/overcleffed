# Overcleffed: AI teammates that keep up with time

Hackathon theme: **time**. Tagline: *The right action at the wrong time is the wrong action.*

## The problem

Decision models like Cloudflare Clef answer typed questions in under a second. We put Clef in two games where timing decides everything: a two-chef kitchen and a growing metro network. Both exposed the same weakness: an instruction that was right a moment ago can be wrong now.

## Act 1: Overcooked (30 s)

- Benchmark: Collab-Overcooked (EMNLP 2025), 30 recipes, two chefs on opposite sides of a wall.
- Setup: code plans the recipe, Clef picks each chef's next skill. 30/30 tasks with a scripted planner.
- Failure found: with Gemma as planner, old goals stayed cached after they were done. The assistant fetched a second egg and picked a delivered egg back up; the chef put a cooked egg down instead of serving.
- Fix: goals now expire. Each tick, completed goals retire before preconditions are checked, and a failed model reply falls back to the deterministic planner, logged as fallback.
- Proof: 7 regression tests fail before, pass after. Real boiled-egg run served in 26 ticks.

## Act 2: Mini Metro (90 s)

Simulator: open-source `python_mini_metro` (pinned commit). Passengers spawn faster over time; a station waiting too long ends the game. The simulation keeps running while Clef thinks, so slow or stale answers cost passengers.

Three controllers, same seeds, same 5 s decision cadence, 3 s deadline, no hidden fallback:

- **Heuristic:** simple hand-written rule.
- **Clef alone:** one big question, "which action is best?"
- **Tree Clef:** code builds a pruned decision tree; Clef answers small observable questions (which station is most urgent, what kind of fix, which legal action), all in one request.

Results at high pressure (spawn rate 900, 120 s, passengers delivered):

| Seed | Heuristic | Clef alone | Tree Clef |
|---:|---:|---:|---:|
| 1 | 27 | 27 | 29 |
| 2 | 33 | 33 | 33 |
| 3 | 26 | 18, game over | 25 |
| 4 | 28 | 7, game over | 27 |
| 5 | 30 | 25 | 30 |

Clef alone collapsed on 2 of 5 seeds. Tree Clef had no game overs and delivered 144 passengers in total vs the heuristic's 144.

## The insight

Clef's own model card: 98.5% on tool-calling choices, 24.7% on ChessBench. Decision models choose well when the situation is visible, and plan ahead poorly. So don't ask them to plan. Let code build and prune the options; let the model answer small, concrete questions it is good at.

## Honest limits

- 5 seeds, one run each. Tree Clef matches the heuristic; it does not beat it yet.
- Structured game state and route operations, not pixel play.
- Our Clef runs on an A16 at about 0.8 s per decision; Cloudflare reports 39 ms for Clef-flash on their hardware.

## Next

1. Look-ahead: simulate the chosen action for a few seconds before applying it (simulator already supports save/restore).
2. Faster hosted Clef to cut stale decisions.
3. Fine-tune Clef on decisions labelled by simulator outcomes.

## Artifacts

- Kitchen reel: `runs/video/server_render/demo_reel_server.mp4`
- Mini Metro side-by-side (seed 4, Clef alone vs tree Clef): path in `docs/mini-metro-spike.md`
- Decision log: `docs/decisions.md`; research: `docs/mini-metro-*.md`
