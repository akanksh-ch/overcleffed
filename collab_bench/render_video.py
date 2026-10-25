"""Render a run_clef.py result JSON to a smooth MP4 (or GIF): pixel-art kitchen, food icons,
chefs gliding between tiles, and each chef's planner goal + Clef probabilities.

Needs a run logged with per-step "pos" and "objects" (run_clef.py adds them).
uv run --with pillow python collab_bench/render_video.py runs/collab_hier2/boiled_egg.json -o out.mp4
Food icons: Twemoji (CC-BY 4.0). Chef/terrain sprites: overcooked_ai (MIT).
"""
import json
import os
import subprocess
import sys
from argparse import ArgumentParser

from PIL import Image, ImageDraw

import render_gif as G  # fonts, colors, sprites, recipe chain

CELL, PAD, PANEL_W = G.CELL, G.PAD, G.PANEL_W
FOOD = os.path.join(G.HERE, "sprites", "food")
BASE = {"potato": "1f954", "carrot": "1f955", "onion": "1f9c5", "egg": "1f95a", "mushroom": "1f344",
        "sweet_potato": "1f360", "taro": "1f360", "corn": "1f33d", "bell_pepper": "1fad1", "eggplant": "1f346",
        "tomato": "1f345", "broccoli": "1f966", "cauliflower": "1f966", "peanut": "1f95c", "coconut": "1f965",
        "bean": "1fad8", "lentil": "1fad8", "chickpea": "1fad8", "green_bean": "1fadb", "pea": "1fadb",
        "green_pea": "1fadb", "romaine_lettuce": "1f96c", "spinach": "1f96c", "chicken": "1f357",
        "beef": "1f969", "pumpkin": "1f383", "zucchini": "1f952", "tofu": "1f9c8", "dish": "1f37d"}
BADGE = {"slices": "1f52a", "baked": "1f525", "boiled": "1f4a7", "mashed": "1f963"}
ICONS = {}


def png(code, size):
    k = (code, size)
    if k not in ICONS:
        ICONS[k] = Image.open(f"{FOOD}/{code}.png").convert("RGBA").resize((size, size), Image.LANCZOS)
    return ICONS[k]


def icon(name, size):
    """Food icon for any recipe item: base ingredient + small badge for how it was processed."""
    if name.endswith(("_soup", "_stew")):
        return png("1f372", size)
    if name.endswith("_patty"):
        return png("1f9c6", size)
    words = name.split("_")
    badge = next((BADGE[w] for w in words if w in BADGE), None)
    core = "_".join(w for w in words if w not in BADGE)
    base = BASE.get(core) or next((BASE[k] for k in sorted(BASE, key=len, reverse=True) if k in core), "1f35e")
    im = png(base, size).copy()
    if badge:
        b = png(badge, size // 2)
        im.alpha_composite(b, (size - size // 2, size - size // 2))
    return im


def terrain(step):
    rows = G.parse_map(step["map"])
    return [[c[0] if c[0] in "XPOCBIDS" else " " for c in r] for r in rows]


def facing(step):
    out = {}
    for y, r in enumerate(G.parse_map(step["map"])):
        for x, c in enumerate(r):
            if c[0] in G.DIRS and c[1:2] in ("0", "1"):
                out[int(c[1])] = G.DIRS[c[0]]
    return out


def draw(step, prev, frac, latest, res, label, done, items, reached, horizon):
    grid = terrain(step)
    w, h = max(len(r) for r in grid), len(grid)
    W = PAD * 3 + w * CELL + PANEL_W
    H = max(PAD * 2 + 70 + h * CELL + 150, 2 * 250 + 110)
    img = Image.new("RGBA", (W, H), G.BG)
    d = ImageDraw.Draw(img)
    F = G.F
    d.text((PAD, PAD), G.short(res["order"].replace("_", " ").title(), 40), fill=G.INK, font=F["title"])
    d.text((PAD, PAD + 34), f"{label}   |   {'SERVED' if done else 'tick ' + str(step['t'])}",
           fill="#a3be8c" if done else G.MUTED, font=F["b"])
    ox, oy = PAD, PAD + 70
    for y, row in enumerate(grid):
        for x, k in enumerate(row):
            x0, y0 = ox + x * CELL, oy + y * CELL
            img.paste(G.sprite("terrain", G.TERRAIN.get(k, "floor.png")), (x0, y0))
            if k in "COB":
                color, name = G.STATION[k]
                d.rounded_rectangle([x0 + 10, y0 + 10, x0 + CELL - 10, y0 + CELL - 10], 12, fill=color,
                                    outline=G.BG, width=3)
            if k in "PSDICOB":
                d.rounded_rectangle([x0 + 14, y0 + CELL - 20, x0 + CELL - 14, y0 + CELL - 4], 6, fill=G.BG)
                d.text((x0 + CELL // 2, y0 + CELL - 12), G.STATION[k][1], fill=G.INK, font=F["s"], anchor="mm")
    # items on counters and inside utensils
    held_pos = {tuple(p) for p in step.get("pos", [])}
    for name, pos, contents, cook in step.get("objects", []):
        x0, y0 = ox + pos[0] * CELL, oy + pos[1] * CELL
        if tuple(pos) in held_pos:
            continue
        shown = [contents] if isinstance(contents, str) else (contents or [name])
        n = len(shown)
        sz = 44 if n == 1 else 34
        for j, it in enumerate(shown):
            img.alpha_composite(icon(it, sz), (x0 + CELL // 2 - sz // 2 + (j - (n - 1) / 2).__round__() * 30,
                                               y0 + CELL // 2 - sz // 2 - 6))
        if cook:  # progress ring while the utensil works
            d.arc([x0 + 6, y0 + 6, x0 + CELL - 6, y0 + CELL - 6], -90, -90 + 360 * min(cook, 3) / 3,
                  fill="#ebcb8b", width=6)
    # chefs, tweened between previous and current tile
    face = facing(step)
    for i in range(2):
        cur = step.get("pos", [[0, 0], [0, 0]])[i]
        old = (prev or step).get("pos", [cur, cur])[i]
        fx = old[0] + (cur[0] - old[0]) * frac
        fy = old[1] + (cur[1] - old[1]) * frac
        x0, y0 = int(ox + fx * CELL), int(oy + fy * CELL)
        dn = face.get(i, "SOUTH")
        held = step["held"][i]
        body = G.sprite("chefs", f"{dn}-dish.png" if held and "dish" in held else f"{dn}.png")
        hat = G.sprite("chefs", f"{dn}-{G.HAT[i]}.png")
        img.alpha_composite(body, (x0, y0))
        img.alpha_composite(hat, (x0, y0))
        if held and held != "dish":
            d.ellipse([x0 + CELL - 46, y0 + 2, x0 + CELL - 2, y0 + 46], fill=G.INK, outline=G.CHEF[i], width=3)
            img.alpha_composite(icon(held, 34), (x0 + CELL - 41, y0 + 7))
    # progress bar + recipe chain with icons
    by = oy + h * CELL + 14
    d.rounded_rectangle([ox, by, ox + w * CELL, by + 8], 4, fill=G.CARD)
    d.rounded_rectangle([ox, by, ox + max(8, int(w * CELL * min(1, (step["t"] + frac) / horizon))), by + 8], 4,
                        fill="#a3be8c" if done else "#5e81ac")
    cy = by + 22
    d.text((ox, cy), "RECIPE CHAIN", fill=G.MUTED, font=F["s"])
    x, cy = ox, cy + 18
    for it in items:
        t = it.replace("_", " ")
        tw = int(d.textlength(t, font=F["s"])) + 40
        if x + tw > ox + w * CELL:
            x, cy = ox, cy + 30
        got = it in reached
        d.rounded_rectangle([x, cy, x + tw, cy + 26], 13, fill="#a3be8c" if got else G.CARD)
        img.alpha_composite(icon(it, 20), (x + 4, cy + 3))
        d.text((x + 28, cy + 13), t, fill=G.BG if got else G.MUTED, font=F["s"], anchor="lm")
        x += tw + 6
    if done:
        cx, cyy = ox + w * CELL // 2, oy + h * CELL // 2
        d.rounded_rectangle([ox + 30, cyy - 40, ox + w * CELL - 30, cyy + 40], 18, fill="#a3be8c")
        img.alpha_composite(icon(res["order"], 56), (ox + 44, cyy - 28))
        d.text((cx + 30, cyy), f"SERVED in {step['t'] + 1} ticks", fill=G.BG, font=F["h"], anchor="mm")
    # decision panels
    px = PAD * 2 + w * CELL
    for i in range(2):
        y = PAD + i * (H - PAD) // 2
        d.rounded_rectangle([px, y, W - PAD, y + (H - PAD) // 2 - PAD], 16, fill=G.CARD)
        img.alpha_composite(G.sprite("chefs", f"SOUTH-{G.HAT[i]}.png").resize((40, 40), Image.NEAREST), (px + 10, y + 4))
        img.alpha_composite(G.sprite("chefs", "SOUTH.png").resize((40, 40), Image.NEAREST), (px + 10, y + 4))
        d.text((px + 56, y + 13), G.NAMES[i], fill=G.INK, font=F["h"])
        held = step["held"][i]
        d.text((px + 170, y + 16), "holds:", fill=G.MUTED, font=F["b"])
        if held:
            img.alpha_composite(icon(held, 24), (px + 222, y + 12))
            d.text((px + 252, y + 16), G.short(held.replace("_", " "), 24), fill=G.INK, font=F["b"])
        else:
            d.text((px + 222, y + 16), "nothing", fill=G.MUTED, font=F["b"])
        goal = (step.get("goal") or [None, None])[i]
        d.text((px + 16, y + 52), "PLANNER GOAL", fill=G.MUTED, font=F["s"])
        d.text((px + 16, y + 70), G.short(goal or "idle", 62), fill="#ebcb8b", font=F["b"])
        dec = latest[i]
        d.text((px + 16, y + 102), "CLEF DECISION" + (f"  ({len(dec['options'])} options)" if dec else ""),
               fill=G.MUTED, font=F["s"])
        if not dec:
            continue
        for j, (name, p) in enumerate(sorted(dec["probs"].items(), key=lambda kv: -kv[1])[:G.TOP_K]):
            yy = y + 124 + j * 26
            chosen = name == dec["choice"]
            d.rounded_rectangle([px + 16, yy, px + 216, yy + 18], 6, fill="#2e3440")
            if p > 0.005:
                d.rounded_rectangle([px + 16, yy, px + 16 + max(8, int(p * 200)), yy + 18], 6,
                                    fill=G.CHEF[i] if chosen else "#5a6272")
            d.text((px + 228, yy + 1), f"{p:4.0%}  {G.short(name, 40)}", fill=G.INK if chosen else G.MUTED, font=F["b"])
    return img.convert("RGB")


def main():
    ap = ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-o", required=True, help=".mp4 or .gif")
    ap.add_argument("--sub", type=int, default=6, help="frames per tick (tweening)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--label", default="HTN planner + Clef-flash")
    ap.add_argument("--horizon", type=int, default=120)
    a = ap.parse_args()
    res = json.load(open(a.json))
    items = G.chain(res["order"], json.load(open(os.path.join(G.HERE, "recipes.json"))))
    decs = sorted(res["decisions"], key=lambda d: d["t"])
    traj, latest, reached, di, frames = res["trajectory"], [None, None], set(), 0, []
    for n, step in enumerate(traj):
        while di < len(decs) and decs[di]["t"] <= step["t"]:
            latest[decs[di]["agent"]] = decs[di]
            di += 1
        reached |= {x for x in step["held"] if x} | {o[2] if isinstance(o[2], str) else o[0]
                                                      for o in step.get("objects", [])}
        done = res["success"] and n == len(traj) - 1
        if done:
            reached.add(res["order"])
        prev = traj[n - 1] if n else None
        for k in range(a.sub):
            frames.append(draw(step, prev, (k + 1) / a.sub, latest, res, a.label, done, items, reached, a.horizon))
    frames += [frames[-1]] * a.fps * 2
    if a.o.endswith(".gif"):
        fr = [f.convert("P", palette=Image.ADAPTIVE, colors=200) for f in frames]
        fr[0].save(a.o, save_all=True, append_images=fr[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    else:
        W, H = frames[0].size
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.o], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
    print(f"{a.o}: {len(frames)} frames, success={res['success']}")


if __name__ == "__main__":
    sys.exit(main())
