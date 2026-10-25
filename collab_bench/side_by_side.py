"""Stack two GIFs vertically, frame-synced; the shorter one holds its last frame.

uv run --with pillow python collab_bench/side_by_side.py top.gif bottom.gif -o out.gif
"""
from argparse import ArgumentParser

from PIL import Image, ImageSequence

ap = ArgumentParser()
ap.add_argument("top")
ap.add_argument("bottom")
ap.add_argument("-o", required=True)
ap.add_argument("--ms", type=int, default=300)
a = ap.parse_args()
A = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(a.top))]
B = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(a.bottom))]
n = max(len(A), len(B))
w, h = max(A[0].width, B[0].width), A[0].height + B[0].height
out = []
for i in range(n):
    f = Image.new("RGB", (w, h), "#14161b")
    f.paste(A[min(i, len(A) - 1)], (0, 0))
    f.paste(B[min(i, len(B) - 1)], (0, A[0].height))
    out.append(f.convert("P", palette=Image.ADAPTIVE, colors=128))
out[0].save(a.o, save_all=True, append_images=out[1:], duration=a.ms, loop=0, optimize=True)
print(f"{a.o}: {n} frames")
