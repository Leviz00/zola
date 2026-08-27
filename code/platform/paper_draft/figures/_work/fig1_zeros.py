"""Figure 1 (fig:zeros): three zero-generating mechanisms schematic."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/mnt/agents/output/paper_draft/figures/fig1_zero_mechanisms.png"

C_STRUCT = "#7a8fa6"   # muted blue-gray  (existence / structural)
C_TECH = "#b08968"     # muted tan        (measurement / technical)
C_SAMP = "#6d9b7e"     # muted green      (sampling)
C_OBS = "#4a4a4a"
TXT = "#222222"

W, H = 10.6, 3.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

lanes = [
    dict(y=2.75, color=C_STRUCT, tag="Existence layer",
         title="Structural zero",
         formula=r"$Z_{ij}\sim\mathrm{Bernoulli}(\pi_{ij})$",
         desc=r"$Z_{ij}=0$: taxon absent from the niche;"
              "\nno sequencing depth can recover it",
         prob=r"miss w.p. $1-\pi_{ij}$"),
    dict(y=1.78, color=C_TECH, tag="Measurement layer",
         title="Technical zero",
         formula=r"detection efficiency $\rho_{ij}$, batch $B_i$",
         desc=r"$Z_{ij}=1$ but the pipeline loses the taxon"
              "\n(extraction, primers, batch effects)",
         prob=r"miss w.p. $1-\rho_{ij}$"),
    dict(y=0.81, color=C_SAMP, tag="Sampling layer",
         title="Sampling zero",
         formula=r"capture $1-g(N_i;\bar\theta_j,\phi)$",
         desc=r"$Z_{ij}=1$, $\rho_{ij}=1$, but finite depth $N_i$"
              "\nfails to capture a rare taxon",
         prob=r"miss w.p. $g(N_i;\bar\theta_j,\phi)$"),
]

for L in lanes:
    y = L["y"]; c = L["color"]
    box = FancyBboxPatch((0.12, y - 0.33), 2.85, 0.60,
                         boxstyle="round,pad=0.035", fc="white",
                         ec=c, lw=1.4)
    ax.add_patch(box)
    ax.text(0.32, y + 0.335, L["tag"], fontsize=7.0, color=c,
            fontweight="bold", va="center",
            bbox=dict(fc="white", ec="none", pad=1.2))
    ax.text(0.30, y + 0.09, L["title"], fontsize=8.8, color=TXT,
            fontweight="bold", va="center")
    ax.text(0.30, y - 0.14, L["formula"], fontsize=7.6, color=TXT, va="center")
    ax.text(3.22, y, L["desc"], fontsize=7.3, color=TXT, va="center")
    ax.text(6.28, y + 0.335, L["prob"], fontsize=7.4, color=c, va="center",
            ha="center", style="italic")
    ar = FancyArrowPatch((7.35, y), (8.55, 1.78), arrowstyle="-|>",
                         mutation_scale=13, lw=1.3, color=c)
    ax.add_patch(ar)

# icons
# structural: presence grid with one absent cell
gx, gy = 5.55, 2.70
for r in range(2):
    for cidx in range(4):
        filled = (r == 0 or cidx < 3)
        ax.add_patch(Rectangle((gx + 0.16 * cidx, gy + 0.10 - 0.18 * r),
                               0.13, 0.13,
                               fc=C_STRUCT if filled else "white",
                               ec=C_STRUCT, lw=0.8))
# technical: efficiency bar
bx, by = 5.85, 1.78
ax.add_patch(Rectangle((bx, by - 0.06), 0.80, 0.13, fc="white", ec=C_TECH, lw=0.9))
ax.add_patch(Rectangle((bx, by - 0.06), 0.50, 0.13, fc=C_TECH, ec="none", alpha=0.9))
ax.text(bx + 0.40, by + 0.16, r"$\rho_{ij}<1$", fontsize=6.6, ha="center", color=C_TECH)
# sampling: mini detection curve inset
sx, sy = 5.85, 0.42
axs = fig.add_axes([sx / W, sy / H, 0.085, 0.15])
xx = np.logspace(2, 6, 60)
g = (5e4 / (5e4 + xx)) ** 0.8
axs.plot(np.log10(xx), 1 - g, color=C_SAMP, lw=1.4)
axs.axvline(np.log10(3e3), color="#999999", lw=0.8, ls=":")
axs.set_xticks([]); axs.set_yticks([])
for s in axs.spines.values():
    s.set_color("#aaaaaa"); s.set_linewidth(0.7)
axs.set_xlabel(r"$\log N_i$", fontsize=5.8, labelpad=0.8)
axs.set_ylabel(r"$1-g$", fontsize=5.8, labelpad=0.8)

# observed node
obs = FancyBboxPatch((8.58, 1.40), 1.32, 0.78, boxstyle="round,pad=0.045",
                     fc="#efefef", ec=C_OBS, lw=1.5)
ax.add_patch(obs)
ax.text(9.24, 1.99, "observed", fontsize=8.2, ha="center", color=C_OBS,
        fontweight="bold")
ax.text(9.24, 1.75, r"$Y_{ij}=0$", fontsize=10.5, ha="center", color=TXT)
ax.text(9.24, 1.54, "only the conjunction\nis observed", fontsize=6.2,
        ha="center", color=C_OBS)
ax.text(9.24, 1.10,
        r"$\Pr(Y_{ij}>0\mid N_i)=\pi_{ij}\,\rho_{ij}\,[1-g(N_i;\bar\theta_j,\phi)]$",
        fontsize=7.4, ha="center", color=TXT)

fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT)
