"""Render a run JSONL into a GIF: game frame + per-chef prob bars + score/latency/entropy timeline.

Log has no state, so we re-simulate: env is deterministic given layout + actions.
Layout is read from <log>.summary.json (args.layout) or --layout.
Usage: SDL_VIDEODRIVER=dummy uv run python viz/replay.py runs/x.jsonl -o out.gif [--ep 0] [--budget 0.2] [--every 2]
"""
import argparse, json, math, os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
from PIL import Image, ImageDraw
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

NAMES = ["up", "down", "left", "right", "stay", "interact"]
ENV_ACT = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0), "interact": "interact"}
COL = [(60, 120, 220), (40, 160, 90)]
W_PANEL, H_TL = 340, 110


def load(path, ep):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ep", 0) == ep:
            rows.append(r)
    return sorted(rows, key=lambda r: r.get("t", 0))


def norm_probs(p):
    """-> dict name->prob or None. Tolerates dict, list aligned to NAMES, or None."""
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        p = dict(zip(NAMES, p))
    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


def entropy(p):
    return -sum(v * math.log2(v) for v in p.values() if v > 0) if p else None


def bars(d, x, y, w, probs, chosen, color, label):
    d.text((x, y), label, fill=color)
    ent = entropy(probs)
    if ent is not None:
        d.text((x + 70, y), f"H={ent:.2f} bits", fill=(80, 80, 80))
    for i, n in enumerate(NAMES):
        yy = y + 16 + i * 16
        d.text((x, yy), n, fill=(0, 0, 0))
        if probs:
            p = probs.get(n, 0)
            d.rectangle([x + 55, yy + 2, x + 55 + int((w - 100) * p), yy + 12],
                        fill=color if n == chosen else (190, 190, 190))
            d.text((x + w - 40, yy), f"{p:.2f}", fill=(0, 0, 0))
        elif n == chosen:
            d.text((x + 55, yy), "<- (no probs logged)", fill=color)


def timeline(d, x, y, w, h, rows, i, budget):
    d.rectangle([x, y, x + w, y + h], outline=(150, 150, 150))
    n = len(rows)
    sx = lambda k: x + int(w * k / max(n - 1, 1))
    lat = [r.get("latency_s") or 0 for r in rows]
    top = max(max(lat), budget * 1.2, 1e-6)
    for k, r in enumerate(rows):  # latency bars; red when late
        bh = int(h * lat[k] / top)
        late = r.get("late") or (lat[k] > budget)
        d.line([sx(k), y + h, sx(k), y + h - bh], fill=(220, 40, 40) if late else (140, 180, 140))
    by = y + h - int(h * budget / top)
    d.line([x, by, x + w, by], fill=(0, 0, 0))
    sc = [r.get("score", 0) for r in rows]
    smax = max(max(sc), 1)
    d.line([(sx(k), y + h - int(h * s / smax)) for k, s in enumerate(sc)], fill=(30, 30, 200), width=2)
    d.line([sx(i), y, sx(i), y + h], fill=(255, 140, 0), width=2)
    d.text((x + 3, y + 2), f"blue=score(max {smax})  green/red=latency (red late)  black={budget}s budget", fill=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("-o", default="viz/out.gif")
    ap.add_argument("--ep", type=int, default=0); ap.add_argument("--layout")
    ap.add_argument("--budget", type=float, default=0.2, help="tick budget seconds")
    ap.add_argument("--every", type=int, default=2, help="render every Nth tick"); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.log, a.ep)
    if a.max: rows = rows[:a.max]
    if not rows: sys.exit("no rows")
    layout, horizon = a.layout, max(len(rows), 400)
    sp = a.log.rsplit(".", 1)[0] + ".summary.json"
    if os.path.exists(sp):
        args = json.load(open(sp)).get("args", {})
        layout = layout or args.get("layout"); horizon = args.get("horizon", horizon)
    layout = layout or "cramped_room"
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()
    viz = StateVisualizer()
    frames = []
    for i, r in enumerate(rows):
        probs = [norm_probs(p) for p in (r.get("probs") or [None, None])]
        if i % a.every == 0:
            surf = viz.render_state(env.state, mdp.terrain_mtx,
                                    hud_data=StateVisualizer.default_hud_data(env.state, score=r.get("score", 0)))
            g = Image.fromarray(np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)))
            gw, gh = g.size
            im = Image.new("RGB", (gw + W_PANEL, max(gh, 260) + H_TL), "white")
            im.paste(g, (0, 0))
            d = ImageDraw.Draw(im)
            d.text((gw + 10, 4), f"t={r['t']}  score={r.get('score', 0)}  lat={(r.get('latency_s') or 0)*1000:.0f}ms"
                   + ("  LATE" if r.get("late") else ""), fill=(200, 0, 0) if r.get("late") else (0, 0, 0))
            for c in (0, 1):
                bars(d, gw + 10, 24 + c * 125, W_PANEL - 20, probs[c], r["actions"][c], COL[c], f"chef {c}")
            timeline(d, 5, im.height - H_TL + 5, im.width - 10, H_TL - 10, rows, i, a.budget)
            frames.append(im)
        env.step(tuple(ENV_ACT[n] for n in r["actions"]))
    frames[0].save(a.o, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f"{a.o}: {len(frames)} frames")


if __name__ == "__main__":
    main()
