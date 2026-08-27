"""Figure 7.2 (fig:real-phi): division of labor of phi, 3 panels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "/mnt/agents/output/"
PC = BASE + "realdata/phi_count_check/results/"
OUT = BASE + "paper_draft/figures/fig72_phi_division.png"

C_DET, C_CNT, C_LB = "#4c72b0", "#55a868", "#8a6d3b"

# ---------------- data ----------------
# (a) subsets
mbqc_phi, mbqc_se = 1454.0, 0.058  # fit_mbqc_summary (1453.585, 0.058)
ci = (mbqc_phi * np.exp(-1.96 * mbqc_se), mbqc_phi * np.exp(1.96 * mbqc_se))
prof = pd.read_csv(BASE + "realdata/karstens/results/dilution_phi_profile.csv")
llmax = prof["profile_loglik"].max()
karstens_lb = float(prof.loc[prof["profile_loglik"] >= llmax - 1.92, "phi"].min())
print("MBQC 95% CI:", [round(v) for v in ci], " Karstens grid LB:", karstens_lb)

# (b) paired per-taxon
pr = pd.read_csv(PC + "compare_phi_paired.csv")
rho = spearmanr(np.log10(pr["phi"]), np.log10(pr["phi_det"]))[0]
med_c, med_d = pr["phi"].median(), pr["phi_det"].median()
print("paired n", len(pr), "spearman", round(rho, 3), "medians",
      round(med_c, 1), round(med_d, 1))

# (c) batch strata
bs = pd.read_csv(PC + "batch_strata_summary.csv")
bs["name"] = bs["stratum"].str.replace("HL:", "", regex=False)
bs["layer"] = pd.Categorical(bs["layer"], ["HL", "BL"], ordered=True)
bs = bs.sort_values(["layer", "phi_det"]).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4),
                         gridspec_kw={"width_ratios": [1.0, 1.05, 1.5]})

# ---------------- (a) phi across subsets ----------------
ax = axes[0]
labels = ["MBQC\nbiological", "MBQC\nfecal mock", "MBQC\noral mock",
          "Karstens\ndilution"]
# interior estimate
ax.errorbar([0], [mbqc_phi], yerr=[[mbqc_phi - ci[0]], [ci[1] - mbqc_phi]],
            fmt="o", color=C_DET, ms=6, capsize=4, lw=1.4, zorder=3)
ax.annotate(f"$\\hat\\phi$=1,454\n95% CI [{ci[0]:,.0f}, {ci[1]:,.0f}]",
            xy=(0.05, mbqc_phi*0.75), xytext=(0.16, 0.13),
            textcoords="axes fraction", fontsize=7, color=C_DET)
# lower bounds
for x, lb in [(1, 1e6), (2, 1e6), (3, karstens_lb)]:
    ax.annotate("", xy=(x, lb), xytext=(x, lb / 6),
                arrowprops=dict(arrowstyle="-|>", color=C_LB, lw=1.6))
    ax.plot([x], [lb], marker="_", ms=14, color=C_LB, mew=2.2)
ax.text(1.5, 1.6e6, r"both mocks: $\phi\geq10^{6}$ (grid top)", fontsize=6.6,
        ha="center", color=C_LB)
ax.text(3, karstens_lb * 1.55, r"$\phi\gtrsim1.6\times10^{5}$" "\n(profile 95% CI)",
        fontsize=6.6, ha="center", color=C_LB)
ax.set_yscale("log"); ax.set_ylim(3e2, 1.3e7)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.6)
ax.set_ylabel(r"detection-side $\hat\phi$ (log scale)", fontsize=8.5)
ax.set_title("(a) $\\hat\\phi$ across MBQC subsets and Karstens", fontsize=8.5)
ax.tick_params(labelsize=7.5)

# ---------------- (b) count vs detection per-taxon phi ----------------
ax = axes[1]
ax.scatter(pr["phi_det"], pr["phi"], s=13, color=C_DET, alpha=0.6,
           edgecolor="none")
lim = (0.5, 2e6)
ax.plot(lim, lim, color="#666666", lw=0.9, ls=":")
ax.axvline(med_d, color=C_DET, lw=1.0, ls="--", alpha=0.8)
ax.axhline(med_c, color=C_CNT, lw=1.0, ls="--", alpha=0.8)
ax.text(2.2e4, med_c * 1.3, f"count median {med_c:.1f}", fontsize=6.8, color=C_CNT)
ax.text(med_d * 0.62, 1.15e6, f"detection median {med_d:,.0f}", fontsize=6.8,
        color=C_DET, rotation=90, va="top", ha="right")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel(r"detection-side $\hat\phi_j$ (log)", fontsize=8.5)
ax.set_ylabel(r"count-side $\hat\phi_j$ (ZIBB, log)", fontsize=8.5)
ax.set_title(f"(b) 96 jointly identifiable taxa, Spearman {rho:.3f}",
             fontsize=8.5)
ax.tick_params(labelsize=7.5)

# ---------------- (c) detection-side phi by stratum ----------------
ax = axes[2]
for i, r in bs.iterrows():
    col = C_DET if r["layer"] == "HL" else C_CNT
    if r["phi_det"] >= 1e5 - 1:
        ax.annotate("", xy=(i, 1.35e5), xytext=(i, 2.6e4),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3))
        ax.plot([i], [1e5], marker="_", ms=11, color=col, mew=1.8)
    else:
        lo = r["phi_det"] * np.exp(-r["se_gamma_det"])
        hi = r["phi_det"] * np.exp(r["se_gamma_det"])
        ax.errorbar([i], [r["phi_det"]], yerr=[[r["phi_det"] - lo],
                    [hi - r["phi_det"]]], fmt="o", ms=5, color=col,
                    capsize=2.5, lw=1.2)
ax.axhline(1454, color="#8a6d3b", lw=1.3, ls="--")
ax.text(0.1, 1150, r"pooled $\hat\phi$=1,454", fontsize=7.2,
        color="#8a6d3b", ha="left")
# layer divider
n_hl = int((bs["layer"] == "HL").sum())
ax.axvline(n_hl - 0.5, color="#bbbbbb", lw=0.9)
ax.text((n_hl - 1) / 2, 4.5e5, "handling labs", fontsize=7.5, ha="center",
        color=C_DET)
ax.text(n_hl + 1.6, 4.5e5, "bioinformatics", fontsize=7.5,
        ha="center", color=C_CNT)
ax.set_yscale("log"); ax.set_ylim(4e2, 8e5)
ax.set_xticks(range(len(bs)))
ax.set_xticklabels(bs["name"], rotation=90, fontsize=6.3)
ax.set_ylabel(r"within-stratum detection $\hat\phi$ (log)", fontsize=8.5)
ax.set_title("(c) MBQC batch strata: interior (dot $\\pm$SE) vs boundary ($\\uparrow$)",
             fontsize=8.5)
ax.tick_params(labelsize=7.5)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
