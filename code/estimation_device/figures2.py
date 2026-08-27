"""figures2.py — 论文 §7 第二批图（任务 3 补）：F3 检出拟合、F4 mock 后验。

F3 `F3_detection_fit.png`：检出率-深度实证分箱 vs 拟合曲线叠加。
  (a) IBDMDB 活检（代表类群：可识别高频/低频/脊区各一，沿用 figures.py 选法）；
  (b) MBQC fecal mock（Table 1 已知成员中最高计数 3 属，对数深度轴）。
F4 `F4_mbqc_mock_posterior.png`：零来源后验区分能力。
  行 = fecal/oral mock；左列 = 零细胞后验分布（present vs absent 属），
  右列 = 属级 π̂（score）分组 strip + AUC 标注。
依赖：mock_blank_analysis.py 与 identifiability.py 的 results 产物。
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import g_closed  # noqa: E402

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
C_P, C_A = "#1b9e77", "#d95f02"


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # ---- (a) IBDMDB
    df = pd.read_csv(RES / "identifiability_ibdmdb.csv")
    z = np.load(DATA / "ibdmdb_genus.npz", allow_pickle=True)
    Y, dep, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
    phi = pd.read_csv(RES / "fit_ibdmdb_summary.csv").iloc[0]["phi_hat"]
    ident = df[df["zone"] == "identifiable"].sort_values("prevalence")
    reps = [ident.iloc[-1], ident.iloc[max(len(ident) // 10, 0)]]
    ridge = df[df["zone"] == "ridge"].sort_values("e_j")
    if len(ridge):
        reps.append(ridge.iloc[len(ridge) // 2])
    ax = axes[0]
    Ngrid = np.logspace(np.log10(dep.min()), np.log10(dep.max()), 200)
    edges = np.unique(np.quantile(np.log10(dep), np.linspace(0, 1, 9)))
    binc = pd.cut(np.log10(dep), bins=edges)
    cols = ["#4c72b0", "#55a868", "#c44e52"]
    for c, r in zip(cols, reps):
        j = int(np.where(taxa == r["taxon"])[0][0])
        det = (Y[:, j] > 0).astype(float)
        emp = pd.Series(det).groupby(binc, observed=True).mean()
        nmid = np.array([np.median(dep[binc == cc]) for cc in emp.index])
        ax.scatter(nmid, emp.to_numpy(), s=30, color=c, zorder=3)
        ax.plot(Ngrid, r["pi_hat"] * (1 - g_closed(Ngrid, r["theta_hat"], phi)),
                color=c, lw=1.6, label=f"{r['taxon']} ({r['zone']})")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("sequencing depth $N$ (log)")
    ax.set_ylabel("detection rate")
    ax.set_title("(a) IBDMDB biopsy (n=178): empirical vs fitted", fontsize=10)
    ax.legend(fontsize=7.5)
    # ---- (b) MBQC fecal mock
    cu = np.load(RES / "mbqc_mock_depthcurve.npz", allow_pickle=True)
    ax = axes[1]
    cents = cu["fecal_centers"]; emp = cu["fecal_emp"]; theo = cu["fecal_theo"]
    reps_m = [str(t) for t in cu["fecal_rep"]]
    for k, (t, c) in enumerate(zip(reps_m, cols)):
        ax.scatter(cents, emp[:, k], s=30, color=c, zorder=3)
        ax.plot(cents, theo[:, k], color=c, lw=1.6,
                label=f"{t} (Table 1 member)")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("sequencing depth $N$ (log)")
    ax.set_title("(b) MBQC fecal mock (n=1,158): known members", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.suptitle("F3. Empirical detection rate vs depth with fitted "
                 r"$\hat\pi_j[1-g(N)]$ curves", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F3_detection_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("F3 done")


def fig4():
    pc = np.load(RES / "mbqc_mock_posterior_cells.npz")
    auc = pd.read_csv(RES / "mbqc_mock_auc.csv").set_index("mock")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, tag in enumerate(["fecal", "oral"]):
        a = auc.loc[tag]
        sp, sa = pc[f"{tag}_present"], pc[f"{tag}_absent"]
        ax = axes[row, 0]
        bins = np.linspace(0, 1, 41)
        ax.hist(sa, bins=bins, density=True, alpha=0.65, color=C_A,
                label=f"absent genera zeros (n cells={a['n_zero_cells_absent']:.3g})")
        ax.hist(sp, bins=bins, density=True, alpha=0.65, color=C_P,
                label=f"present genera zeros (n cells={a['n_zero_cells_present']:.3g})")
        ax.set_xlabel(r"$P(Z_{ij}=0\mid Y_{ij}=0,N_i)$ (structural-zero posterior)")
        ax.set_ylabel("density")
        ax.set_title(f"{tag} mock — zero-source posterior by group\n"
                     f"cell-level AUC={a['auc_cell_posterior']:.3f}", fontsize=9.5)
        ax.legend(fontsize=7.5)
        # 属级 π̂ strip
        d = pd.read_csv(RES / f"mbqc_mock_fit_{tag}.csv")
        ax2 = axes[row, 1]
        rng = np.random.default_rng(7)
        for x, msk, col, lab in ((0, d["known_present"].to_numpy(), C_P, "Table 1 present"),
                                 (1, ~d["known_present"].to_numpy(), C_A, "absent")):
            ax2.scatter(rng.normal(x, 0.05, int(msk.sum())),
                        d.loc[msk, "pi_hat"], s=11, alpha=0.55, color=col,
                        edgecolor="none", label=lab)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([f"present\n(n={int(d['known_present'].sum())})",
                             f"absent\n(n={int((~d['known_present']).sum())})"])
        ax2.set_ylabel(r"genus-level $\hat\pi_j$")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title(f"{tag} mock — genus scores\n"
                      f"AUC($\\hat\\pi$)={a['auc_pi']:.3f}", fontsize=9.5)
        ax2.legend(fontsize=7.5, markerscale=1.6)
    fig.suptitle("F4. MBQC mock truth validation: structural-zero posterior "
                 "and presence scores", y=0.99, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F4_mbqc_mock_posterior.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("F4 done")


if __name__ == "__main__":
    fig3()
    fig4()
