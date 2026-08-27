"""make_image3.py — English redraw of media/image-3.png (3915x1637).
Identifiability of (pi, theta, phi) from the detection-depth curve.
Curves use the closed-form g(N;theta,phi) from code/estimation/model.py."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import g_closed  # noqa: E402

W, H, DPI = 3915, 1637, 150
BROWN = "#8D4C35"
TEAL = "#3E6B6E"
TEAL_LINE = "#3C686A"
OLIVE = "#A08C5B"
OLIVE_CURVE = "#818161"
DOT = "#3B3B3B"
GRAY666 = "#666666"
DARK = "#333333"
GRID = "#F0F0F0"
SPINE = "#D0D0D0"
EMDASH = "\u2014"

def D(N, pi, th, phi):
    return pi * (1.0 - g_closed(N, th, phi))

Ns = np.logspace(0, 6.4, 800)
XMAX = 10 ** 6.4

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")

FS_SU = 26      # suptitle
FS_TI = 25      # panel titles
FS_TK = 17      # tick labels
FS_LB = 19      # axis labels
FS_AN = 15.5    # annotations
FS_FO = 20      # formula
FS_LG = 16      # legend

axA = fig.add_axes([0.0484, 0.1017, 0.4320, 0.7117])
axB = fig.add_axes([0.5669, 0.1017, 0.4320, 0.7117])

for ax in (axA, axB):
    ax.set_xscale("log")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([i / 5 for i in range(6)])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) / 10,
                                          numticks=70))
    ax.tick_params(which="major", length=10, width=1.8, color="black",
                   labelsize=FS_TK)
    ax.tick_params(which="minor", length=6, width=1.4, color="black")
    ax.tick_params(which="both", axis="y", length=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(2.0)
    ax.grid(True, which="major", color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    ax.set_xlabel("Sequencing depth N (log scale)", fontsize=FS_LB,
                  color="#222222", labelpad=14)
    ax.set_ylabel("Detection probability", fontsize=FS_LB, color="#222222",
                  labelpad=14)

fig.text(0.5, 0.968, "Identifying parameters from the detection" + EMDASH + "depth curve",
         ha="center", va="center", fontsize=FS_SU, color="black")

# ---------------- Panel A ----------------
axA.set_title("A. One curve, three pieces of information", loc="left",
              fontsize=FS_TI, color="black", pad=16)
axA.plot(Ns, D(Ns, 0.8, 5e-4, 3e3), color=BROWN, lw=3.4, zorder=5)
axA.axhline(0.8, color=GRAY666, lw=2.2, ls=(0, (1.2, 2.2)), zorder=3)
axA.axvline(9000, color=TEAL_LINE, lw=2.6, ls=(0, (7, 4.5)), zorder=4)

axA.text(0.033, 0.955, r"$D(N) = \pi \cdot [1 - g(N;\, \theta, \varphi)]$",
         transform=axA.transAxes, ha="left", va="top", fontsize=FS_FO,
         color=DARK, zorder=6)

# plateau annotation
axA.text(0.627, 0.935, r"Plateau height = $\pi$ (presence probability)",
         transform=axA.transAxes, ha="left", va="top", fontsize=FS_AN - 1,
         color=GRAY666, zorder=6)
axA.annotate("", xy=(0.866, 0.805), xytext=(0.807, 0.895),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="#777777", lw=2.0,
                             shrinkA=0, shrinkB=0), zorder=6)

# early-rise annotation
axA.text(0.0756, 0.300,
         r"Early rise (N $\ll$ $\varphi$): driven by the product" + "\n" +
         r"$a$ = $\varphi\theta$, $\varphi$ absorbed",
         transform=axA.transAxes, ha="left", va="top", fontsize=FS_AN,
         color=BROWN, zorder=6, linespacing=1.45)
axA.annotate("", xy=(200, 0.066), xytext=(0.183, 0.200),
             xycoords=("data", "data"), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=BROWN, lw=2.0,
                             shrinkA=0, shrinkB=0), zorder=6)

# transition-zone annotation
axA.text(0.532, 0.250,
         r"Transition zone N $\approx$ 3$\varphi$: curve still rising," + "\n" +
         r"its shape is sensitive to $\varphi$ " + EMDASH + r" this pins $\varphi$",
         transform=axA.transAxes, ha="left", va="top", fontsize=FS_AN,
         color=TEAL, zorder=6, linespacing=1.45)
axA.annotate("", xy=(9000, 0.70), xytext=(0.665, 0.205),
             xycoords=("data", "data"), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.0,
                             shrinkA=0, shrinkB=0), zorder=6)

# ---------------- Panel B ----------------
axB.set_title("B. Why some parameters are separable, others not", loc="left",
              fontsize=FS_TI, color="black", pad=16)
axB.plot(Ns, D(Ns, 0.8, 5e-4, 3e3), color=BROWN, lw=3.4, zorder=5,
         label=r"$(\pi, \theta, \varphi) = (0.8, 5\times10^{-4}, 3\times10^{3})$")
axB.plot(Ns, D(Ns, 0.8, 5e-5, 3e4), color=TEAL, lw=3.4, zorder=5,
         ls=(0, (9, 5)),
         label=r"$(0.8, 5\times10^{-5}, 3\times10^{4})$, same product $a = 1.5$")
axB.plot(Ns, D(Ns, 0.8, 7e-6, 3e3), color=OLIVE_CURVE, lw=3.0, zorder=5,
         label=r"$(0.8, 7\times10^{-6}, 3\times10^{3})$")
axB.plot(Ns, D(Ns, 0.4, 1.4e-5, 3e3), color=DOT, lw=3.0, zorder=5,
         ls=(0, (1.2, 1.6)),
         label=r"$(0.4, 1.4\times10^{-5}, 3\times10^{3})$, same $\pi\theta$")
leg = axB.legend(loc="upper left", bbox_to_anchor=(0.018, 0.99),
                 fontsize=FS_LG, frameon=False, handlelength=3.2,
                 labelspacing=0.75, borderaxespad=0)
for t in leg.get_texts():
    t.set_color("#1A1A1A")

axB.axvline(9000, color=TEAL_LINE, lw=2.6, ls=(0, (7, 4.5)), zorder=4)
axB.text(0.588, 0.948,
         r"Prop. 1(iii): the depth range must straddle" + "\n" +
         r"N $\approx$ 3$\varphi$ to separate $\varphi$" + "\n" +
         r"from the product $a$ = $\varphi\theta$",
         transform=axB.transAxes, ha="left", va="top", fontsize=FS_AN,
         color=TEAL, zorder=6, linespacing=1.45)
axB.text(0.029, 0.251,
         r"Ridge (rare taxa): same $\pi\theta$ $\Rightarrow$ curves nearly coincide" + "\n" +
         r"within reachable depths (max difference 6%);" + "\n" +
         r"only $\pi\theta$ identifiable " + EMDASH + " separation would need" + "\n" +
         r"depths $\sim 10^{24}$",
         transform=axB.transAxes, ha="left", va="top", fontsize=FS_AN - 1.5,
         color=OLIVE, zorder=6, linespacing=1.45)
axB.annotate("", xy=(3.6e5, 0.075), xytext=(0.402, 0.138),
             xycoords=("data", "data"), textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=OLIVE, lw=2.0,
                             shrinkA=0, shrinkB=0), zorder=6)

fig.savefig("/mnt/agents/output/zola_en_upload/media/image-3.png",
            dpi=DPI, facecolor="white")
print("saved")
