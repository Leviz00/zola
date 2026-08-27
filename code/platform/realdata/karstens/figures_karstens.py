"""figures_karstens.py — Karstens2019 稀释系列与空白锚定图（任务 A）。

F_K1_dilution_overview.png : 稀释设计 vs 实测（梯子对比 + 属检出数轨迹）
F_K2_detection_gof.png     : 模型 S（经验梯子）逐稀释度观测 vs 期望检出
F_K3_blank_anchor.png      : 空白谱与 π̂ 锚定前后对比
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/mnt/agents/output/realdata/karstens")
RES = OUT / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def f_k1():
    z = np.load(OUT / "data_karstens_genus.npz", allow_pickle=True)
    Y, depths = z["Y"].astype(float), z["depths"].astype(float)
    taxa = z["taxa"].astype(str)
    is_blank = z["is_blank"].astype(bool)
    Ym, Nm = Y[~is_blank], depths[~is_blank]
    emp = pd.read_csv(RES / "dilution_empirical.csv")
    members = ["Escherichia/Shigella", "Pseudomonas", "Salmonella",
               "Lactobacillus", "Listeria", "Staphylococcus",
               "Enterococcus", "Bacillus", "unclassified_Enterobacteriaceae"]
    rel = Ym / Nm[:, None]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(9)
    ax[0].plot(x, emp["nominal_d"], "o-", label="nominal $3^{-k}$")
    ax[0].plot(x, emp["empirical_d_core_reads"], "s-",
               label="empirical (member reads)")
    ax[0].plot(x, emp["dna_conc_d"], "^-", label="DNA concentration")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dilution level k (D0..D8)")
    ax[0].set_ylabel("relative mock abundance factor $d_k$")
    ax[0].set_title("(a) Nominal vs realized dilution ladder")
    ax[0].legend(fontsize=8)

    ax[1].plot(x, emp["core_detected"], "o-", color="tab:blue",
               label="D0-detected genera (of 17)")
    ax[1].plot(x, emp["noncore_detected"], "s-", color="tab:red",
               label="other genera detected (of 182)")
    ax[1].set_xlabel("dilution level k")
    ax[1].set_ylabel("# genera detected")
    ax[1].set_title("(b) Detection vs dilution (genus level)")
    ax[1].legend(fontsize=8)

    for t in members:
        j = int(np.where(taxa == t)[0][0])
        ax[2].plot(x, rel[:, j] * 100, marker="o", ms=3, lw=1, label=t)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("dilution level k")
    ax[2].set_ylabel("relative abundance (%)")
    ax[2].set_title("(c) True mock member trajectories")
    ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "F_K1_dilution_overview.png", dpi=150)
    plt.close(fig)


def f_k2():
    gof = pd.read_csv(RES / "dilution_gof_by_level.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(gof))
    ax[0].plot(x, gof["observed_detected"], "ko-", label="observed", ms=5)
    ax[0].plot(x, gof["expected_S_empirical"], "s--",
               label="model S (empirical ladder)", ms=4)
    ax[0].plot(x, gof["expected_S_flat"], "^--",
               label="model S (no dilution)", ms=4)
    ax[0].plot(x, gof["expected_S_nominal"], "x--",
               label="model S (nominal ladder)", ms=4)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(gof["sample"])
    ax[0].set_ylabel("# D0-detected genera detected (of 17)")
    ax[0].set_title("(a) Observed vs expected detections")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, gof["residual_S_empirical"],
              color=np.where(np.abs(gof["residual_S_empirical"]) > 2,
                             "tab:red", "tab:gray"))
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhline(2, color="tab:red", ls=":", lw=0.8)
    ax[1].axhline(-2, color="tab:red", ls=":", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(gof["sample"])
    ax[1].set_ylabel("binomial z-residual")
    ax[1].set_title("(b) Residuals, model S (empirical ladder)")
    fig.tight_layout()
    fig.savefig(FIG / "F_K2_detection_gof.png", dpi=150)
    plt.close(fig)


def f_k3():
    df = pd.read_csv(RES / "anchor_pertaxon.csv")
    inb = df["blank_reads"] > 0
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(df.loc[inb, "blank_rel_abund"],
                  df.loc[inb, "prevalence_mock"], s=12, alpha=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("blank relative abundance $c_j$")
    ax[0].set_ylabel("prevalence in 9 mock samples")
    ax[0].set_title("(a) Blank spectrum vs mock prevalence\n"
                    "(Spearman 0.405, p=2.8e-9)")
    ax[1].scatter(df.loc[~inb, "pi_hat_U"], df.loc[~inb, "pi_hat_A2"],
                  s=12, alpha=0.5, label="not in blank", color="tab:gray")
    ax[1].scatter(df.loc[inb, "pi_hat_U"], df.loc[inb, "pi_hat_A2"],
                  s=14, alpha=0.7, label="in blank", color="tab:red")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].set_xlabel(r"$\hat\pi_j$ unanchored ($\rho\equiv1$)")
    ax[1].set_ylabel(r"$\hat\pi_j$ blank-anchored (model A2)")
    ax[1].set_title("(b) Presence probability before/after anchoring")
    ax[1].legend(fontsize=8)
    ax[2].scatter(df.loc[inb, "blank_rel_abund"],
                  df.loc[inb, "pi_shrinkage_A2"], s=12, alpha=0.6)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("blank relative abundance $c_j$")
    ax[2].set_ylabel(r"$\hat\pi$ shrinkage (U $-$ A2)")
    ax[2].set_title("(c) Shrinkage vs blank abundance")
    fig.tight_layout()
    fig.savefig(FIG / "F_K3_blank_anchor.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    f_k1()
    f_k2()
    f_k3()
    print("figures written to", FIG)
