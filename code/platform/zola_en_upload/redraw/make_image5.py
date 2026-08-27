"""make_image5.py — English redraw of media/image-5.png (947x327).
Empirical FDR bars; ZOLA value changed to 0.047 per user request."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

W, H, DPI = 947, 327, 100
BG = "#FAF7F2"
RED = "#9E4A3A"
GREEN = "#4F7A5A"
DARK = "#2C2823"

methods = ["DESeq2", "TSS+Wilcoxon", "LinDA", "ANCOM-BC2", "ZOLA (ours)"]
vals = [0.483, 0.304, 0.272, 0.201, 0.047]
colors = [RED, RED, RED, RED, GREEN]

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0.0517, 0.1024, 0.943, 0.9557 - 0.1024])
ax.set_facecolor(BG)
ax.set_xlim(-0.6, 4.55)
ax.set_ylim(0, 0.55)
ax.set_yticks([i / 10 for i in range(6)])
ax.set_yticklabels([f"{i / 10:.1f}" for i in range(6)], fontsize=13, color=DARK)
ax.set_xticks(range(5))
ax.set_xticklabels(methods, fontsize=13, color=DARK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

bars = ax.bar(range(5), vals, width=0.577, color=colors, zorder=2)

# value labels
for i, v in enumerate(vals):
    ytxt = v + 0.012 if v > 0.05 else 0.058
    ax.text(i, ytxt, f"{v:.3f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=DARK, zorder=3)

# nominal 5% ceiling
ax.axhline(0.05, color=RED, lw=1.8, ls=(0, (8, 6)), zorder=3)
t = ax.text(-0.43, 0.072, "Nominal ceiling 5%", ha="left", va="center",
            fontsize=11.5, color="#A6502E", zorder=5)
t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=BG)])

fig.text(0.007, (0.1024 + 0.9557) / 2, "Empirical FDR", rotation=90,
         ha="center", va="center", fontsize=12.5, color=DARK)

fig.savefig("/mnt/agents/output/zola_en_upload/media/image-5.png",
            dpi=DPI, facecolor=BG)
print("saved")
