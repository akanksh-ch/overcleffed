"""Render a run_clef.py result JSON to a presentation GIF.

Left: the kitchen with chefs, held items and station icons. Right: for each chef, the
planner's goal and Clef's top options with probabilities (chosen option highlighted).

uv run --with pillow python collab_bench/render_gif.py runs/collab_hier/boiled_egg.json -o out.gif
"""
import json
import sys
from argparse import ArgumentParser

from PIL import Image, ImageDraw, ImageFont

CELL, PAD, PANEL_W, TOP_K = 96, 24, 560, 5
BG, INK, MUTED, CARD = "#14161b", "#eceff4", "#8a93a6", "#1f232b"
FLOOR, WALL, SHARED = "#2a2f3a", "#3b4252", "#4c566a"
STATION = {"P": ("#5e81ac", "POT"), "O": ("#d08770", "OVEN"), "C": ("#a3be8c", "CHOP"),
           "B": ("#88c0d0", "BLEND"), "I": ("#ebcb8b", "INGR"), "D": ("#d8dee9", "DISH"),
           "S": ("#b48ead", "SERVE")}
CHEF = ["#ef5350", "#42a5f5"]  # match the red/blue hats
NAMES = ["Chef", "Assistant"]
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
SPR = {}


def sprite(sheet, name):
    """Crop a 15x15 pixel-art frame from overcooked_ai's sprite sheets, scaled crisp to CELL."""
    key = (sheet, name)
    if key not in SPR:
        meta = json.load(open(f"{HERE}/sprites/{sheet}.json"))["frames"][name]["frame"]
        img = Image.open(f"{HERE}/sprites/{sheet}.png").convert("RGBA")
        SPR[key] = img.crop((meta["x"], meta["y"], meta["x"] + meta["w"], meta["y"] + meta["h"])).resize(
            (CELL, CELL), Image.NEAREST)
    return SPR[key]


DIRS = {"\u2191": "NORTH", "\u2193": "SOUTH", "\u2190": "WEST", "\u2192": "EAST"}
TERRAIN = {"X": "counter.png", "P": "pot.png", "S": "serve.png", "D": "dishes.png", "I": "onions.png"}
HAT = ["redhat", "bluehat"]
ARROW = {"\u2191": (0, -1), "\u2193": (0, 1), "\u2190": (-1, 0), "\u2192": (1, 0)}


def font(n, bold=False):
    for f in ["/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]:
        try:
            ft = ImageFont.truetype(f, n)
            if bold and f.endswith("[wght].ttf"):
                ft.set_variation_by_name("Bold")
            return ft
        except OSError:
            pass
    return ImageFont.load_default()


F = {"title": font(26, True), "h": font(18, True), "b": font(15), "s": font(13), "tile": font(13, True)}


def parse_map(s):
    rows = [r for r in s.split("\n") if r.strip()]
    return [[r[i:i + 8].strip() or " " for i in range(0, len(r.rstrip()), 8)] for r in rows]


def short(x, n):
    return x if len(x) <= n else x[:n - 1] + "\u2026"


def chain(order, recipes):
    """Linear-ish production order of the recipe tree, leaves first."""
    prod = {}
    for u in recipes.values():
        for item, r in u.items():
            prod.setdefault(item, r["recipe"])
    out = []
    def walk(x):
        for y in prod.get(x, []):
            if y != x:
                walk(y)
        if x not in out:
            out.append(x)
    walk(order)
    return out


def seen_items(step):
    s = set(h for h in step["held"] if h)
    return s


def frame(step, latest, res, label, done, items=None, reached=None, horizon=120):
    grid = parse_map(step["map"])
    w, h = max(len(r) for r in grid), len(grid)
    W = PAD * 3 + w * CELL + PANEL_W
    H = max(PAD * 2 + 70 + h * CELL + 150, 2 * 250 + 110)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD), short(res["order"].replace("_", " ").title(), 40), fill=INK, font=F["title"])
    status = "SERVED" if done else f"tick {step['t']}"
    d.text((PAD, PAD + 34), f"{label}   \u2022   {status}", fill="#a3be8c" if done else MUTED, font=F["b"])
    ox, oy = PAD, PAD + 70
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            x0, y0 = ox + x * CELL, oy + y * CELL
            k = c[0]
            img.paste(sprite("terrain", TERRAIN.get(k, "floor.png")), (x0, y0))
            if k in ("C", "B", "O"):  # no sprite in overcooked_ai: colored station on a counter
                color, name = STATION[k]
                d.rounded_rectangle([x0 + 14, y0 + 14, x0 + CELL - 14, y0 + CELL - 14], 10, fill=color,
                                    outline=BG, width=3)
                d.text((x0 + CELL // 2, y0 + CELL // 2), name, fill=BG, font=F["tile"], anchor="mm")
            if k in ("P", "S", "D", "I"):
                name = STATION[k][1]
                d.rounded_rectangle([x0 + 18, y0 + CELL - 20, x0 + CELL - 18, y0 + CELL - 4], 6, fill=BG)
                d.text((x0 + CELL // 2, y0 + CELL - 12), name, fill=INK, font=F["s"], anchor="mm")
            if k in DIRS and c[1:2] in ("0", "1"):
                i = int(c[1])
                dname = DIRS[k]
                held = step["held"][i]
                body = f"{dname}-dish.png" if held and ("dish" in held) else f"{dname}.png"
                img.paste(sprite("chefs", body), (x0, y0), sprite("chefs", body))
                hat = sprite("chefs", f"{dname}-{HAT[i]}.png")
                img.paste(hat, (x0, y0), hat)
                if held:
                    t = short(held.replace("_", " "), 14)
                    d.rounded_rectangle([x0 + 2, y0 + 2, x0 + CELL - 2, y0 + 20], 6, fill=CHEF[i])
                    d.text((x0 + CELL // 2, y0 + 11), t, fill=BG, font=F["s"], anchor="mm")
    d.text((ox, oy + h * CELL + 10), "left: assistant side   middle: shared counter   right: chef side",
           fill=MUTED, font=F["s"])
    by = oy + h * CELL + 34
    d.rounded_rectangle([ox, by, ox + w * CELL, by + 8], 4, fill=CARD)
    d.rounded_rectangle([ox, by, ox + max(8, int(w * CELL * min(1, (step["t"] + 1) / horizon))), by + 8], 4,
                        fill="#a3be8c" if done else "#5e81ac")
    if items:
        cy = by + 22
        d.text((ox, cy), "RECIPE CHAIN", fill=MUTED, font=F["s"])
        x = ox
        cy += 18
        for k, it in enumerate(items):
            t = it.replace("_", " ")
            tw = int(d.textlength(t, font=F["s"])) + 16
            if x + tw > ox + w * CELL:
                x, cy = ox, cy + 26
            got = it in reached
            d.rounded_rectangle([x, cy, x + tw, cy + 20], 10, fill="#a3be8c" if got else CARD)
            d.text((x + tw // 2, cy + 10), t, fill=BG if got else MUTED, font=F["s"], anchor="mm")
            x += tw + 6
    if done:
        d.rounded_rectangle([ox + 40, oy + h * CELL // 2 - 34, ox + w * CELL - 40, oy + h * CELL // 2 + 34], 18,
                            fill="#a3be8c")
        d.text((ox + w * CELL // 2, oy + h * CELL // 2), f"SERVED in {step['t'] + 1} ticks", fill=BG,
               font=F["title"], anchor="mm")
    px = PAD * 2 + w * CELL
    for i in range(2):
        y = PAD + i * (H - PAD) // 2
        d.rounded_rectangle([px, y, W - PAD, y + (H - PAD) // 2 - PAD], 16, fill=CARD)
        d.ellipse([px + 16, y + 16, px + 32, y + 32], fill=CHEF[i])
        d.text((px + 42, y + 13), NAMES[i], fill=INK, font=F["h"])
        held = (step["held"][i] or "nothing").replace("_", " ")
        d.text((px + 160, y + 16), f"holds: {short(held, 28)}", fill=MUTED, font=F["b"])
        goal = (step.get("goal") or [None, None])[i]
        d.text((px + 16, y + 46), "PLANNER GOAL", fill=MUTED, font=F["s"])
        d.text((px + 16, y + 64), short(goal or "idle", 62), fill="#ebcb8b", font=F["b"])
        dec = latest[i]
        d.text((px + 16, y + 96), "CLEF DECISION" + (f"  (t={dec['t']}, {len(dec['options'])} options)" if dec else ""),
               fill=MUTED, font=F["s"])
        if not dec:
            continue
        top = sorted(dec["probs"].items(), key=lambda kv: -kv[1])[:TOP_K]
        for j, (name, p) in enumerate(top):
            yy = y + 118 + j * 26
            chosen = name == dec["choice"]
            d.rounded_rectangle([px + 16, yy, px + 216, yy + 18], 6, fill="#2e3440")
            if p > 0.005:
                d.rounded_rectangle([px + 16, yy, px + 16 + max(8, int(p * 200)), yy + 18], 6,
                                    fill=CHEF[i] if chosen else "#5a6272")
            d.text((px + 228, yy + 1), f"{p:4.0%}  {short(name, 40)}", fill=INK if chosen else MUTED, font=F["b"])
    return img


def main():
    ap = ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-o", required=True)
    ap.add_argument("--ms", type=int, default=350, help="ms per frame")
    ap.add_argument("--label", default="HTN planner + Clef-flash")
    ap.add_argument("--horizon", type=int, default=120)
    a = ap.parse_args()
    res = json.load(open(a.json))
    import os
    items = chain(res["order"], json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes.json"))))
    reached = set()
    latest, frames, di = [None, None], [], 0
    decs = sorted(res["decisions"], key=lambda d: d["t"])
    traj = res["trajectory"]
    for n, step in enumerate(traj):
        while di < len(decs) and decs[di]["t"] <= step["t"]:
            latest[decs[di]["agent"]] = decs[di]
            di += 1
        reached |= seen_items(step) | {x for u in step.get("utensils", {}).values() for x in u}
        done = res["success"] and n == len(traj) - 1
        if done:
            reached.add(res["order"])
        frames.append(frame(step, latest, res, a.label, done, items, reached, a.horizon))
    frames += [frames[-1]] * 8
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    frames[0].save(a.o, save_all=True, append_images=frames[1:], duration=a.ms, loop=0, optimize=True)
    print(f"{a.o}: {len(frames)} frames, success={res['success']}")


if __name__ == "__main__":
    sys.exit(main())
