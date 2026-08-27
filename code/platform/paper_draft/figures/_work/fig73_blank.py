"""Figure 7.3 (fig:real-blank): blank spectra and rho anchoring, 3 panels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "/mnt/agents/output/"
KR = BASE + "realdata/karstens/results/"
OUT = BASE + "paper_draft/figures/fig73_blank_anchor.png"

C_ALL, C_MOCK, C_CORE, C_NON = "#4c72b0", "#c44e52", "#4c72b0", "#c44e52"

mb = pd.read_csv(BASE + "realdata/results/mbqc_blank_contamination.csv")
de = pd.read_csv(KR + "dilution_empirical.csv")
an = pd.read_csv(KR + "anchor_pertaxon.csv")

rho_f = spearmanr(mb["blank_prevalence"], mb["fecal_mock_prevalence"])[0]
print("spearman blank-fecal:", round(rho_f, 3))

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35))

# ---------------- (a) blank vs fecal-mock prevalence ----------------
ax = axes[0]
mem = mb["is_table1_member"].to_numpy()
rng = np.random.default_rng(5)
jx = rng.normal(0, 0.006, len(mb)); jy = rng.normal(0, 0.006, len(mb))
ax.scatter(mb.loc[~mem, "fecal_mock_prevalence"] + jx[~mem],
           mb.loc[~mem, "blank_prevalence"] + jy[~mem], s=8, color=C_ALL,
           alpha=0.4, edgecolor="none", rasterized=True)
ax.scatter(mb.loc[mem, "fecal_mock_prevalence"] + jx[mem],
           mb.loc[mem, "blank_prevalence"] + jy[mem], s=16, color=C_MOCK,
           alpha=0.85, edgecolor="white", lw=0.3, zorder=3)
ax.plot([-0.02, 1.02], [-0.02, 1.02], color="#888888", lw=0.8, ls=":")
ax.set_xlabel("fecal-mock prevalence", fontsize=8.5)
ax.set_ylabel("blank prevalence (401 blanks)", fontsize=8.5)
ax.set_title(f"(a) MBQC blanks mirror mock spectra\n"
             f"Spearman {rho_f:.3f} (717 genera)", fontsize=8.5)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C_ALL,
                              alpha=0.5, label="other genera"),
                   plt.Line2D([], [], marker="o", ls="", color=C_MOCK,
                              label="Table-1 mock members")],
          fontsize=7, loc="lower right", markerscale=1.5, framealpha=0.9)
ax.tick_params(labelsize=7.5)

# ---------------- (b) Karstens detected genera per dilution ----------------
ax = axes[1]
x = de["level"]
ax.plot(x, de["core_detected"], "o-", color=C_CORE, lw=1.5, ms=4.5,
        label="core mock members (of 17)")
ax.plot(x, de["noncore_detected"], "s-", color=C_NON, lw=1.5, ms=4.5,
        label="carryover / other genera (of 182)")
ax.set_xlabel("dilution level (D0..D8)", fontsize=8.5)
ax.set_ylabel("genera detected", fontsize=8.5)
ax.set_xticks(range(9)); ax.set_ylim(-4, 130)
ax2 = ax.twinx()
ax2.plot(x, de["nominal_d"], color="#888888", lw=1.1, ls="--")
ax2.set_yscale("log"); ax2.set_ylim(2e-5, 3)
ax2.set_ylabel(r"nominal ladder $3^{-k}$ (gray, log)", fontsize=7.5,
               color="#666666")
ax2.tick_params(labelsize=6.5, colors="#666666")
ax.set_title("(b) Karstens series: detection rises\nwith nominal dilution",
             fontsize=8.5)
ax.legend(fontsize=7, loc="center left", framealpha=0.9)
ax.tick_params(labelsize=7.5)

# ---------------- (c) blank anchoring A2 vs U ----------------
ax = axes[2]
inb = an["blank_rel_abund"] > 0
ax.scatter(an.loc[~inb, "pi_hat_U"], an.loc[~inb, "pi_hat_A2"], s=11,
           color="#999999", alpha=0.6, edgecolor="none",
           label=f"not in blank (n={int((~inb).sum())})")
ax.scatter(an.loc[inb, "pi_hat_U"], an.loc[inb, "pi_hat_A2"], s=13,
           color=C_NON, alpha=0.75, edgecolor="none",
           label=f"detected in blank (n={int(inb.sum())})")
ax.plot([0, 1.02], [0, 1.02], color="#666666", lw=0.9, ls=":")
med_u = an.loc[inb, "pi_hat_U"].median(); med_a = an.loc[inb, "pi_hat_A2"].median()
ax.annotate(f"median $\\hat\\pi_j$: {med_u:.3f}$\\to${med_a:.3f}",
            xy=(med_u, med_a), xytext=(0.30, 0.72), fontsize=7.2,
            color=C_NON,
            arrowprops=dict(arrowstyle="->", color=C_NON, lw=0.9))
ax.set_xlabel(r"$\hat\pi_j$ unanchored (model U, $\rho\equiv1$)", fontsize=8.5)
ax.set_ylabel(r"$\hat\pi_j$ blank-anchored (model A2)", fontsize=8.5)
ax.set_title("(c) Karstens blank anchoring:\nrandom contamination channel A2 vs U",
             fontsize=8.5)
ax.legend(fontsize=7, loc="lower right", markerscale=1.5, framealpha=0.9)
ax.set_xlim(-0.03, 1.06); ax.set_ylim(-0.03, 1.06)
ax.tick_params(labelsize=7.5)

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
print("medians in-blank", round(med_u, 3), round(med_a, 3),
      "not-in-blank", round(an.loc[~inb, 'pi_hat_U'].median(), 3),
      round(an.loc[~inb, 'pi_hat_A2'].median(), 3))
