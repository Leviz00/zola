"""make_image2.py — English redraw of media/image-2.png (774x615).
Schematic detection-rate vs sequencing-depth curve (S-shape to ceiling)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

W, H, DPI = 774, 615, 100
BG = "#F7F3E8"
BRICK = "#8C3B36"
SPINE = "#969080"
ARROW = "#8A8378"
NOTE = "#6B6259"
AXISLBL = "#857D70"

# centerline of the original curve, traced in pixel coords (y down)
trace = np.array([
    (45, 535), (120, 514), (195, 475), (270, 424.5), (345, 368.5),
    (420, 312.5), (495, 260.5), (570, 217.5), (645, 185), (714, 169.5),
])
f = PchipInterpolator(trace[:, 0], trace[:, 1])
xs = np.linspace(45, 714, 600)
ys = f(xs)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # pixel coords, y down
ax.axis("off")

# spines
ax.plot([44.5, 44.5], [63, 549], color=SPINE, lw=2, solid_capstyle="butt", zorder=2)
ax.plot([44.5, 714], [549, 549], color=SPINE, lw=2, solid_capstyle="butt", zorder=2)

# ceiling dashed line + label
ax.plot([44.5, 714], [167.5, 167.5], color=BRICK, lw=2,
        ls=(0, (4.3, 2.2)), zorder=2)
ax.text(395, 136, r"Ceiling = presence probability $\pi_j$", ha="center",
        va="center", fontsize=15, color=BRICK, zorder=3)

# curve
ax.plot(xs, ys, color=BRICK, lw=3.5, solid_capstyle="round", zorder=3)

# slope arrow + annotations
ax.annotate("", xy=(358, 302), xytext=(275, 382),
            arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.6,
                            shrinkA=0, shrinkB=0), zorder=3)
ax.text(352, 385, r"Slope $\Rightarrow$ abundance $\theta_j$", ha="left",
        va="center", fontsize=14, color=NOTE, zorder=3)
ax.text(352, 435, r"Bend $\Rightarrow$ dispersion $\varphi$ (N $\approx$ 3$\varphi$)",
        ha="left", va="center", fontsize=14, color=NOTE, zorder=3)

# axis labels
ax.text(28, 30, "Detection rate P(Y>0 | N)", ha="left", va="top",
        fontsize=13, color=AXISLBL, zorder=3)
ax.text(705, 585, "Sequencing depth N (log scale)", ha="right", va="center",
        fontsize=13, color=AXISLBL, zorder=3)

fig.savefig("/mnt/agents/output/zola_en_upload/media/image-2.png",
            dpi=DPI, facecolor=BG)
print("saved")
