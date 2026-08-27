"""make_image1.py — English redraw of media/image-1.png (714x459).
Toy ASV count table: 4 samples x 5 taxa, zeros in brick red."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

W, H, DPI = 714, 459, 100
BG = "#F7F3E8"
HDR = "#EFE8D6"
GRID = "#DED6C5"
DARK = "#2C2823"
RED = "#8C3B36"

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # pixel coords, y down
ax.axis("off")

vlines = [14.5, 148, 255, 362.5, 469.5, 577, 684]
hlines = [6.5, 76.5, 147, 216, 286, 356.5]

# fills: header row + first column
ax.add_patch(Rectangle((vlines[0], hlines[0]), vlines[-1] - vlines[0],
                       hlines[1] - hlines[0], facecolor=HDR, edgecolor="none", zorder=1))
ax.add_patch(Rectangle((vlines[0], hlines[1]), vlines[1] - vlines[0],
                       hlines[-1] - hlines[1], facecolor=HDR, edgecolor="none", zorder=1))
# grid
for x in vlines:
    ax.plot([x, x], [hlines[0], hlines[-1]], color=GRID, lw=1.5, zorder=2,
            solid_capstyle="butt")
for y in hlines:
    ax.plot([vlines[0], vlines[-1]], [y, y], color=GRID, lw=1.5, zorder=2,
            solid_capstyle="butt")

def cx(c):  # column center
    return (vlines[c] + vlines[c + 1]) / 2

def cy(r):  # row center
    return (hlines[r] + hlines[r + 1]) / 2

FS_HDR, FS_NUM = 11.5, 11
# header
ax.text(cx(0), cy(0), "Sample\\Taxon",
        ha="center", va="center", fontsize=FS_HDR, fontweight="bold",
        color=DARK, zorder=3)
for c in range(1, 6):
    ax.text(cx(c), cy(0), f"Taxon {c}", ha="center", va="center",
            fontsize=FS_HDR, fontweight="bold", color=DARK, zorder=3)

data = [
    ["152", "0", "37", "0", "6"],
    ["0", "0", "104", "12", "0"],
    ["88", "3", "0", "0", "21"],
    ["0", "45", "0", "7", "0"],
]
for r in range(4):
    ax.text(cx(0), cy(r + 1), f"Sample {r + 1}", ha="center", va="center",
            fontsize=FS_HDR, fontweight="bold", color=DARK, zorder=3)
    for c in range(5):
        v = data[r][c]
        ax.text(cx(c + 1), cy(r + 1), v, ha="center", va="center",
                fontsize=FS_NUM, color=(RED if v == "0" else DARK), zorder=3)

# notes (auto-fit: shrink until the composed second line stays inside W-14)
FS_NOTE = 10.5
while FS_NOTE > 7:
    t1 = ax.text(18, 428, "In real ASV tables ", ha="left", va="center",
                 fontsize=FS_NOTE, color=DARK, zorder=3)
    fig.canvas.draw()
    x2 = t1.get_window_extent().x1
    t2 = ax.text(x2, 428, "60%\u201390% of cells are zeros", ha="left", va="center",
                 fontsize=FS_NOTE, color=RED, fontweight="bold", zorder=3)
    fig.canvas.draw()
    x3 = t2.get_window_extent().x1
    t3 = ax.text(x3, 428, " \u2014 understanding these zeros is understanding the data itself.",
                 ha="left", va="center", fontsize=FS_NOTE, color=DARK, zorder=3)
    fig.canvas.draw()
    if t3.get_window_extent().x1 <= W - 12:
        break
    for t in (t1, t2, t3):
        t.remove()
    FS_NOTE -= 0.5
ax.text(18, 396, "Row = sample, column = taxon; row sums give sequencing depth $N_i$.",
        ha="left", va="center", fontsize=FS_NOTE, color=DARK, zorder=3)

fig.savefig("/mnt/agents/output/zola_en_upload/media/image-1.png",
            dpi=DPI, facecolor=BG)
print("saved")
