"""make_wrap_fig2.py -- redesigned wrap figure: 3 level panels + pooled-delta
panel on its own axis (the paired gains were sub-marker-size on the 0-1 axis)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, INK, MUT = "#C2402F", "#2563EB", "#1a1a1a", "#6b6b6b"
BASE = {"zinq": "A0", "deseq2": "A0", "locom": "A1n", "ldm": "A1n",
        "wilcoxon": "A1", "linda": "A1", "twochannel": "A1"}
NAME = {"twochannel": "two-channel (this paper)", "zinq": "ZINQ",
        "ldm": "LDM", "locom": "LOCOM", "wilcoxon": "TSS+Wilcoxon",
        "linda": "LinDA", "deseq2": "DESeq2"}
ORDER = ["twochannel", "zinq", "ldm", "locom", "wilcoxon", "linda", "deseq2"]
CELLN = {2001: "REAL-PRES", 2002: "REAL-MIX", 2003: "REAL-HARD"}
SUB = {2001: "presence truths", 2002: "mixed truths",
       2003: "intensity truths, $n{=}100$"}
POOLED = {  # method: (dTPR, label, significant at p<0.01)
    "twochannel": (0.023, "11+/0−", True),
    "zinq": (0.020, "10+/0−", True),
    "ldm": (0.027, "7+/0−", True),
    "locom": (0.002, "2+/1−", False),
    "wilcoxon": (0.015, "7+/0−", True),
    "linda": (0.005, "2+/0−", False),
    "deseq2": (0.037, "12+/0−", True),
}

s = pd.concat([pd.read_csv(f"/home/claude/ch_smoke/wrap_{b}_summary.csv")
               for b in ("main", "ldm", "deseq2")], ignore_index=True)


def get(cell, meth, arm, wt, col):
    r = s[(s.cell == cell) & (s.method == meth) & (s.arm == arm) &
          (s.weight == wt)]
    return float(r[col].iloc[0]) if len(r) else np.nan


fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True,
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.95]))
YPOS = {m: len(ORDER) - 1 - i for i, m in enumerate(ORDER)}

for ax, cell in zip(axes[:3], (2001, 2002, 2003)):
    for m in ORDER:
        y = YPOS[m]
        col = RED if m == "twochannel" else BLUE
        if False:
            pass
        else:
            b = get(cell, m, BASE[m], "none", "tpr")
            a = get(cell, m, "A2", "Wdet", "tpr")
            ax.plot([b, a], [y, y], "-", color=col, lw=2, alpha=0.8,
                    zorder=2, solid_capstyle="round")
            ax.plot([b], [y], "o", mfc="white", mec=col, mew=1.6, ms=7.5,
                    zorder=3)
            ax.plot([a], [y], "o", color=col, ms=7.5, zorder=3)
            fdp = get(cell, m, "A2", "Wdet", "fdp")
            ax.text(1.04, y, f"{fdp:.03f}", va="center", ha="left",
                    fontsize=7.5, color=MUT, clip_on=False)
    ax.set_yticks([YPOS[m] for m in ORDER])
    ax.set_yticklabels([NAME[m] for m in ORDER], fontsize=9, color=INK)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("two-channel"):
            lbl.set_fontweight("bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "", "0.5", "", "1"], fontsize=8)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlabel("TPR (union truth)", fontsize=8.5, color=INK)
    ax.set_title(f"{CELLN[cell]}\n{SUB[cell]}", fontsize=9.5, color=INK,
                 pad=10)
    ax.text(1.04, 1.01, "FDP", fontsize=7.5, color=MUT, ha="left",
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, colors=MUT)

# ---- pooled paired-delta panel (its own axis) ----
axd = axes[3]
for m in ORDER:
    y = YPOS[m]
    d, lab, sig = POOLED[m]
    col = RED if m == "twochannel" else BLUE
    axd.plot([0, d], [y, y], "-", color=col, lw=2.4, alpha=0.85, zorder=2,
             solid_capstyle="round")
    axd.plot([d], [y], "o", color=col, ms=7.5, zorder=3)
    star = "*" if sig else ""
    axd.text(d + 0.0022, y, f"+{d:.03f}{star}  ({lab})", va="center",
             ha="left", fontsize=7.3, color=INK, clip_on=False)
axd.axvline(0, color="0.55", lw=1, zorder=1)
axd.set_xlim(-0.003, 0.042)
axd.set_xticks([0, 0.01, 0.02, 0.03])
axd.set_xticklabels(["0", "0.01", "0.02", "0.03"], fontsize=8)
axd.set_ylim(-0.6, len(ORDER) - 0.4)
axd.set_xlabel("pooled paired $\\Delta$TPR", fontsize=8.5, color=INK)
axd.set_title("upstream gain\n(own scale)", fontsize=9.5, color=INK, pad=10)
axd.grid(axis="x", color="0.92", lw=0.7, zorder=0)
axd.spines[["top", "right"]].set_visible(False)
axd.tick_params(length=2.5, colors=MUT)


h1, = axes[0].plot([], [], "o", mfc="white", mec="0.3", mew=1.6,
                   label="unweighted arm")
h2, = axes[0].plot([], [], "o", color="0.3",
                   label="+ upstream weights (W-det)")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.33, -0.055))
fig.text(0.80, -0.028, "* one-sided Wilcoxon $p<0.01$", fontsize=8,
         color=MUT, ha="center")
plt.subplots_adjust(wspace=0.50, bottom=0.20, left=0.13, right=0.94)

for out in ("/home/claude/paper_v2/figures/figW_wrap2.pdf",
            "/home/claude/paper_v2/figures/figW_wrap2.png"):
    fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved")
