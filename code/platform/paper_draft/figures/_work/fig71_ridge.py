"""Figure 7.1 (fig:real-ridge): pi--theta ridge in real data, 3 panels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/agents/output/"
OUT = BASE + "paper_draft/figures/fig71_ridge_realdata.png"

d = pd.read_csv(BASE + "analysis/ej_criterion/results/realdata_pertaxon_scored.csv")
DS = [("ibdmdb", "IBDMDB ($n$=178)"), ("mbqc", "MBQC ($n$=13,562)"),
      ("agp", "AGP ($n$=9,511)")]
C_ID, C_RG = "#4c72b0", "#dd8452"
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
rng = np.random.default_rng(11)

# ---------- (a) theta distribution per dataset ----------
ax = axes[0]
for k, (ds, lab) in enumerate(DS):
    g = d[d["dataset"] == ds]
    y = k + rng.normal(0, 0.09, len(g))
    ident = g["ident_c_info"].to_numpy()
    bnd = g["on_boundary_pi"].to_numpy()
    ax.scatter(g.loc[ident, "theta_hat"], y[ident], s=9, color=C_ID,
               alpha=0.55, edgecolor="none", rasterized=True)
    ax.scatter(g.loc[~ident, "theta_hat"], y[~ident], s=9, color=C_RG,
               alpha=0.6, edgecolor="none", rasterized=True)
    ax.scatter(g.loc[bnd, "theta_hat"], y[bnd], s=22, facecolor="none",
               edgecolor="#222222", lw=0.8, rasterized=True)
ax.set_xscale("log")
ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in DS], fontsize=8)
ax.set_xlabel(r"$\hat{\bar\theta}_j$ (log scale)", fontsize=8.5)
ax.set_title("(a) mean composition by identifiability region", fontsize=8.5)
ax.tick_params(labelsize=7.5)

# ---------- (b) pi vs theta ----------
ax = axes[1]
for ds, lab in DS:
    g = d[d["dataset"] == ds]
    ident = g["ident_c_info"].to_numpy()
    ax.scatter(g.loc[ident, "theta_hat"], g.loc[ident, "pi_hat"], s=9,
               color=C_ID, alpha=0.5, edgecolor="none", rasterized=True)
    ax.scatter(g.loc[~ident, "theta_hat"], g.loc[~ident, "pi_hat"], s=9,
               color=C_RG, alpha=0.6, edgecolor="none", rasterized=True)
ax.axhline(0.9999, color="#8a6d3b", lw=1.2, ls="--")
ax.text(1.5e-6, 0.865, r"box boundary $\hat\pi_j=0.9999$", fontsize=7.2,
        color="#8a6d3b", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
ax.set_xscale("log"); ax.set_ylim(-0.03, 1.06)
ax.set_xlabel(r"$\hat{\bar\theta}_j$ (log scale)", fontsize=8.5)
ax.set_ylabel(r"$\hat\pi_j$", fontsize=8.5)
ax.set_title(r"(b) $\hat\pi_j$ vs $\hat{\bar\theta}_j$: ridge taxa on the boundary", fontsize=8.5)
ax.tick_params(labelsize=7.5)

# ---------- (c) per-taxon I_j ----------
ax = axes[2]
for k, (ds, lab) in enumerate(DS):
    g = d[d["dataset"] == ds]
    y = k + rng.normal(0, 0.09, len(g))
    ident = g["ident_c_info"].to_numpy()
    xj = np.log10(g["I_j"].clip(lower=1e-8))
    ax.scatter(xj[ident], y[ident], s=9, color=C_ID, alpha=0.55,
               edgecolor="none", rasterized=True)
    ax.scatter(xj[~ident], y[~ident], s=9, color=C_RG, alpha=0.6,
               edgecolor="none", rasterized=True)
ax.axvline(0, color="#8a6d3b", lw=1.4, ls="--")
ax.text(0.08, 2.42, r"$I_j=1$", fontsize=8, color="#8a6d3b")
ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in DS], fontsize=8)
ax.set_xlabel(r"profiled Godambe information $I_j$ (log$_{10}$)", fontsize=8.5)
ax.set_title("(c) per-taxon information and the $I_j=1$ cutoff", fontsize=8.5)
ax.set_ylim(-0.55, 2.75)
ax.tick_params(labelsize=7.5)
ax.annotate("IBDMDB/AGP: lower-bound $\hat\phi$ (conservative)",
            xy=(0.5, 0.965), xycoords="axes fraction", fontsize=6.8,
            color="#555555", ha="center")

handles = [plt.Line2D([], [], marker="o", ls="", color=C_ID,
                      label=r"identifiable ($I_j\geq1$, interior)"),
           plt.Line2D([], [], marker="o", ls="", color=C_RG, label="ridge region"),
           plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#222222",
                      label=r"boundary $\hat\pi_j=0.9999$ (panel a)")]
fig.legend(handles=handles, fontsize=7.3, loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.90))
fig.tight_layout(w_pad=1.4, rect=(0.01, 0.02, 1, 0.90))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
