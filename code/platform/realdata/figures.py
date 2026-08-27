"""figures.py — 论文 §7 第一批真实数据图（任务 4，英文标注）。

F1 三数据集测序深度分布（log10 直方图，并排）
F2 三数据集 e_j 分布（log10 轴直方图 + 经验跨界线 e* 与模拟脊区参考线）
F3 π̂ vs θ̄̂ 散点（按可识别区/脊区着色，一个数据集一个 panel）
F4 检出率–深度曲线实证样例（每数据集 3 个代表类群：可识别高频、
   可识别低频、脊区；分箱经验检出率散点 + 拟合曲线叠加）

用法：python3 figures.py            （依赖 results/ 下的 fit/identifiability 输出）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import g_closed  # noqa: E402

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = ["ibdmdb", "mbqc", "agp"]
TITLES = {"ibdmdb": "IBDMDB/HMP2 (n=178)", "mbqc": "MBQC-base (n=13,562)",
          "agp": "AGP (n=9,511)"}
C_IDENT, C_RIDGE = "#1b9e77", "#d95f02"


def load(name):
    df = pd.read_csv(RES / f"identifiability_{name}.csv")
    z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
    return df, z


# ---------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, name in zip(axes, NAMES):
        z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
        d = np.log10(z["depths"].astype(float))
        ax.hist(d, bins=50, color="#4c72b0", edgecolor="white", lw=0.3)
        ax.set_title(TITLES[name], fontsize=10)
        ax.set_xlabel(r"sequencing depth $N_i$ (log$_{10}$)")
        ax.axvline(np.log10(z["depths"].min()), color="grey", ls=":", lw=1)
    axes[0].set_ylabel("number of samples")
    fig.suptitle("F1. Sequencing depth distributions", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F1_depth_distributions.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("F1 done")


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, name in zip(axes, NAMES):
        df, _ = load(name)
        s = pd.read_csv(RES / f"identifiability_{name}_summary.csv").iloc[0]
        le = np.log10(np.maximum(df["e_j"], 1e-12))
        ident = df["zone"] == "identifiable"
        bins = np.linspace(le.min(), le.max(), 40)
        ax.hist(le[ident], bins=bins, color=C_IDENT, alpha=0.75,
                label=f"identifiable ({ident.sum()})")
        ax.hist(le[~ident], bins=bins, color=C_RIDGE, alpha=0.75,
                label=f"ridge ({(~ident).sum()})")
        ax.axvline(np.log10(s["e_crossover"]), color="black", ls="--", lw=1.2,
                   label=f"empirical crossover e*={s['e_crossover']:.2g}")
        ax.axvline(np.log10(0.047), color="purple", ls=":", lw=1.2,
                   label="sim ridge value 0.047")
        ax.set_title(TITLES[name], fontsize=10)
        ax.set_xlabel(r"effective detection strength $e_j$ (log$_{10}$)")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("number of taxa")
    fig.suptitle(r"F2. Distribution of $e_j=\hat\phi\bar\theta_j"
                 r"[\psi(\hat\phi+N_{\min})-\psi(\hat\phi)]$", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F2_ej_distributions.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("F2 done")


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name in zip(axes, NAMES):
        df, _ = load(name)
        ident = df["zone"] == "identifiable"
        for msk, col, lab in ((ident, C_IDENT, "identifiable"),
                              (~ident, C_RIDGE, "ridge")):
            ax.scatter(df.loc[msk, "theta_hat"], df.loc[msk, "pi_hat"],
                       s=14, alpha=0.65, color=col, label=lab,
                       edgecolor="none")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\hat{\bar\theta}_j$ (baseline composition, log)")
        ax.set_ylabel(r"$\hat\pi_j$ (presence probability)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(TITLES[name], fontsize=10)
        ax.legend(fontsize=8, markerscale=1.5)
    fig.suptitle(r"F3. $\hat\pi_j$ vs $\hat{\bar\theta}_j$ by "
                 "identifiability zone", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F3_pi_theta_scatter.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("F3 done")


def _pick_representatives(df):
    """可识别高频 / 可识别低频 / 脊区各 1 个代表类群。"""
    ident = df[df["zone"] == "identifiable"].sort_values("prevalence")
    hi = ident.iloc[-1]
    lo = ident.iloc[max(len(ident) // 10, 0)]      # 低流行率端第 10 百分位
    ridge = df[df["zone"] == "ridge"].sort_values("e_j")
    rg = ridge.iloc[len(ridge) // 2] if len(ridge) else ident.iloc[0]
    return [("identifiable, high prevalence", hi),
            ("identifiable, low prevalence", lo),
            ("ridge zone", rg)]


def fig4():
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row, name in enumerate(NAMES):
        df, z = load(name)
        Y, dep, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
        phi = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]["phi_hat"]
        Ngrid = np.logspace(np.log10(dep.min()), np.log10(dep.max()), 200)
        # 深度分箱（log 等距，8 箱）
        edges = np.quantile(np.log10(dep), np.linspace(0, 1, 9))
        binc = pd.cut(np.log10(dep), bins=np.unique(edges))
        for col, (lab, r) in enumerate(_pick_representatives(df)):
            ax = axes[row, col]
            j = int(np.where(taxa == r["taxon"])[0][0])
            det = (Y[:, j] > 0).astype(float)
            emp = pd.Series(det).groupby(binc, observed=True).mean()
            nmid = np.array([np.median(dep[binc == c]) for c in emp.index])
            ax.scatter(nmid, emp.to_numpy(), s=28, color="#4c72b0",
                       zorder=3, label="empirical (depth-binned)")
            curve = r["pi_hat"] * (1 - g_closed(Ngrid, r["theta_hat"], phi))
            ax.plot(Ngrid, curve, color="#c44e52", lw=1.8,
                    label=r"fitted $\hat\pi_j[1-g(N;\hat{\bar\theta}_j,"
                    r"\hat\phi)]$")
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{TITLES[name]} — {r['taxon']}\n{lab} "
                         f"($e_j$={r['e_j']:.2g})", fontsize=8.5)
            if row == 2:
                ax.set_xlabel("sequencing depth $N$ (log)")
            if col == 0:
                ax.set_ylabel("detection rate")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("F4. Empirical detection-rate vs depth with fitted "
                 "detection curves", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "F4_detection_curves.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("F4 done")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
