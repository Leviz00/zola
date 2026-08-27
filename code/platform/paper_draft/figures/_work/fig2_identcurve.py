"""Figure 2 (fig:identcurve): identifiability curve on the I_j axis."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/agents/output/"
OUT = BASE + "paper_draft/figures/fig2_identifiability_curve.png"

sw = pd.read_csv(BASE + "analysis/ej_criterion/results/sweep_perrep.csv")
sw["abs_err"] = sw["logit_err"].abs()
sw["good"] = (~sw["on_bnd_pi"]) & (sw["abs_err"] <= 2)

panels = [
    ("phiknown", "clean", r"$\phi$ known, clean depths"),
    ("phiknown", "spike", r"$\phi$ known, depth spike (5% $N\leq10$)"),
    ("joint", "clean", r"$\phi$ joint, clean depths"),
]

C_GOOD, C_BAD = "#4c72b0", "#c44e52"
CLIP = 8.0

fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.45), sharey=True)
for ax, (arm, scen, title) in zip(axes, panels):
    d = sw[(sw["arm"] == arm) & (sw["scenario"] == scen)]
    x = np.log10(d["I_j"].clip(lower=1e-8))
    y = d["abs_err"].clip(upper=CLIP)
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(d))
    xa, ya = x.to_numpy()[idx], y.to_numpy()[idx]
    gd = d["good"].to_numpy()[idx]
    ax.scatter(xa[~gd], ya[~gd], s=5, alpha=0.30, color=C_BAD,
               edgecolor="none", rasterized=True,
               label="fail ($|\\Delta\\mathrm{logit}\\,\\pi|>2$ or boundary)")
    ax.scatter(xa[gd], ya[gd], s=5, alpha=0.25, color=C_GOOD,
               edgecolor="none", rasterized=True, label="pass")
    # per-cell median + IQR (8 design cells)
    grp = d.assign(x=x, y=y).groupby("taxon")
    med = grp["y"].median(); q25 = grp["y"].quantile(0.25)
    q75 = grp["y"].quantile(0.75); xm = grp["x"].mean().to_numpy()
    order = np.argsort(xm); xm = xm[order]
    med = med.to_numpy()[order]; q25 = q25.to_numpy()[order]; q75 = q75.to_numpy()[order]
    ax.plot(xm, med, color="#222222", lw=1.6, label="cell median")
    ax.fill_between(xm, q25, q75, color="#222222", alpha=0.15, lw=0)
    ax.axvline(0, color="#8a6d3b", lw=1.4, ls="--")
    ax.text(0.06, CLIP - 0.35, r"$I_j=1$", fontsize=8, color="#8a6d3b")
    ax.axhline(2, color="#888888", lw=0.9, ls=":")
    # fixed-rule specificity within panel
    sel = (d["I_j"] >= 1) & (~d["on_bnd_pi"])
    spec = d.loc[sel, "good"].mean()
    ax.set_title(f"{title}\nspecificity of $I_j\\geq1$ rule = {spec:.3f}",
                 fontsize=8.5)
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(-0.2, CLIP + 0.2)
    ax.tick_params(labelsize=7.5)

axes[0].set_ylabel(r"$|\mathrm{logit}\,\hat\pi_j-\mathrm{logit}\,\pi_j|$"
                   " (clipped at 8)", fontsize=8.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=3,
           markerscale=2.2, frameon=False, bbox_to_anchor=(0.5, 0.925))
fig.text(0.5, 0.015, r"per-taxon profiled Godambe information $I_j$ (log$_{10}$)",
         ha="center", fontsize=9)
fig.tight_layout(w_pad=1.2, rect=(0.02, 0.05, 1, 0.93))
fig.savefig(OUT, dpi=300)
print("saved", OUT)
for arm, scen, _ in panels:
    d = sw[(sw["arm"] == arm) & (sw["scenario"] == scen)]
    sel = (d["I_j"] >= 1) & (~d["on_bnd_pi"])
    print(arm, scen, "n_pass", int(sel.sum()), "spec", round(d.loc[sel, "good"].mean(), 4))
