"""figures_phi.py — 计数侧 vs 检出侧 φ 对照图（任务 B）。

F_P1_count_vs_detection_phi.png : (a) 计数侧 log φ_j 分布 vs 检出侧 1454；
                                  (b) 逐类群配对散点（若 det_pertaxon 存在）
F_P2_batch_strata_phi.png       : 按 HL 实验室 / BL 流程分层的 φ：
                                  (a) 检出侧层 φ̂（±SE）vs 全局 1454；
                                  (b) 计数侧层中位 φ（点）+ 层内 IQR
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path("/mnt/agents/output/realdata/phi_count_check/results")
FIG = Path("/mnt/agents/output/realdata/phi_count_check/figures")
FIG.mkdir(exist_ok=True)
PHI_DET = 1454.0


def f_p1():
    df = pd.read_csv(RES / "count_phi_mbqc_pertaxon.csv")
    inf = df[(df["n_pos"] >= 1000) & (df["se_logphi"] <= 0.5)
             & (~df["phi_on_boundary"])]
    paired = RES / "compare_phi_paired.csv"
    ncols = 2 if paired.exists() else 1
    fig, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 4.2))
    if ncols == 1:
        ax = [ax]
    ax[0].hist(np.log10(inf["phi"]), bins=30, alpha=0.75, color="tab:blue")
    ax[0].axvline(np.log10(PHI_DET), color="tab:red", lw=2,
                  label=r"detection-side shared $\hat\phi$=1454")
    ax[0].axvline(np.log10(inf["phi"].median()), color="tab:blue", ls="--",
                  label=f"count-side median={inf['phi'].median():.3g}")
    ax[0].set_xlabel(r"$\log_{10}\phi$")
    ax[0].set_ylabel("# taxa (informative subset)")
    ax[0].set_title("(a) Count-side per-taxon ZIBB $\\phi_j$ distribution")
    ax[0].legend(fontsize=8)
    if ncols == 2:
        m = pd.read_csv(paired)
        ax[1].scatter(np.log10(m["phi_det"]), np.log10(m["phi"]), s=12,
                      alpha=0.6)
        lo = min(np.log10(m["phi_det"]).min(), np.log10(m["phi"]).min())
        hi = max(np.log10(m["phi_det"]).max(), np.log10(m["phi"]).max())
        ax[1].plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax[1].set_xlabel(r"detection-side $\log_{10}\phi_j$ (per-taxon)")
        ax[1].set_ylabel(r"count-side $\log_{10}\phi_j$ (ZIBB)")
        ax[1].set_title("(b) Per-taxon pairing, both information sources")
    fig.tight_layout()
    fig.savefig(FIG / "F_P1_count_vs_detection_phi.png", dpi=150)
    plt.close(fig)


def f_p2():
    sdf = pd.read_csv(RES / "batch_strata_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, layer, title in ((ax[0], "HL", "(a) by handling lab (HL)"),
                            (ax[1], "BL", "(b) by bioinformatics pipeline (BL)")):
        sub = sdf[sdf["layer"] == layer].sort_values("phi_det")
        x = np.arange(len(sub))
        a.errorbar(x, sub["phi_det"], yerr=sub["phi_det"] * sub["se_gamma_det"],
                   fmt="o", ms=5, color="tab:red",
                   label="detection-side $\\hat\\phi$ per stratum (±SE)")
        a.errorbar(x, sub["count_median_phi"],
                   yerr=[sub["count_median_phi"] - sub["count_q25_phi"],
                         sub["count_q75_phi"] - sub["count_median_phi"]],
                   fmt="s", ms=5, color="tab:blue",
                   label="count-side median $\\phi_j$ (IQR)")
        a.axhline(PHI_DET, color="tab:red", ls=":", lw=1,
                  label="global detection $\\hat\\phi$=1454")
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels([s.split(":")[1] for s in sub["stratum"]],
                          rotation=60, fontsize=7)
        a.set_title(title)
        a.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "F_P2_batch_strata_phi.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    f_p1()
    import os
    if os.path.exists(RES / "batch_strata_summary.csv"):
        f_p2()
    print("figures ->", FIG)
