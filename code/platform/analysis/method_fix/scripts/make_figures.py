"""make_figures.py — method_fix 对比图（300dpi，低饱和暖色系，Unicode 中文）。

(a) fig_a_zero_posterior_v1_vs_v2.png：4 格 × (v1 / v2最优λ) 的 4×2 面板，
    每面板两条 jitter 点带（上=结构零 红，下=抽样零 绿）+ 四分位刻度，
    画法同 /mnt/agents/output/zero_posterior_vs_oracle.png。
(b) fig_b_metric_bars.png：每格 v1 vs v2 的 PR-AUC / Brier / φ回收 分组条形
    （reps 0-2 均值，误差线=逐 rep 点）。
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/mnt/agents/output/analysis/method_fix"
NPZ_DIR = os.path.join(BASE, "npz")
SUM_CSV = os.path.join(BASE, "dev_eval_summary.csv")

C_STRUCT = "#C0574B"   # 低饱和暖红
C_SAMP = "#7A9B7E"     # 低饱和绿
C_V1 = "#C9A26B"       # 暖沙
C_V2 = "#A65E52"       # 暖砖
CELL_ORDER = [6, 2, 11, 22]
CELL_TITLES = {
    6: "格 6 · Beta二项 · 非informative · sz=0",
    2: "格 2 · Beta二项 · informative · sz=0.3",
    11: "格 11 · 三层 · 非informative · sz=0.3",
    22: "格 22 · 三层 · 非informative · sz=0.1",
}


def load_npz(cell, rep, arm_tag):
    hits = glob.glob(os.path.join(NPZ_DIR, f"cell{cell:02d}_rep{rep}_{arm_tag}*.npz"))
    hits = [h for h in hits if not h.endswith(".tmp.npz")]
    if not hits:
        return None
    return np.load(sorted(hits)[0])


def strip_ax(ax, z, title):
    s, lab = z["scores"], z["labels"].astype(int)
    rng = np.random.default_rng(0)
    for val, color, yc in ((1, C_STRUCT, 1.0), (0, C_SAMP, 0.0)):
        x = s[lab == val]
        y = yc + rng.uniform(-0.16, 0.16, x.size)
        ax.scatter(x, y, s=3, color=color, alpha=0.25, linewidths=0,
                   rasterized=True)
        qs = np.quantile(x, [0.25, 0.5, 0.75])
        for q in qs:
            ax.plot([q, q], [yc - 0.14, yc + 0.14], color="#333333", lw=1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["结构零", "抽样零"], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.45, 1.45)
    auc = float(pd.Series([np.nan]).iloc[0]) if False else None
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def fig_a(sel):
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for i, cid in enumerate(CELL_ORDER):
        lam = sel[cid]
        for k, (arm_tag, arm_lab) in enumerate(
                [("v1", "v1 旧管线"), (f"v2_l{lam:g}", f"v2 λ={lam:g}")]):
            z = load_npz(cid, 0, arm_tag)
            ax = axes[i, k]
            if z is None:
                ax.text(0.5, 0.5, "缺运行", ha="center")
                continue
            s, lab = z["scores"], z["labels"]
            from summarize import auc_mw
            auc = auc_mw(lab, s)
            boundary = "是" if (bool(z["cnt_phi_on_boundary"])
                                or bool(z["cov_phi_on_boundary"])) else "否"
            title = (f"{CELL_TITLES[cid]} · {arm_lab}\n"
                     f"后验 AUC={auc:.2f} · 零共 {s.size:,} 个"
                     f"（结构 {lab.mean():.0%}）· φ 撞界={boundary}")
            strip_ax(ax, z, title)
    for ax in axes[-1, :]:
        ax.set_xlabel("ZOLA 后验 P(此零为结构零 | 数据), 0-1")
    fig.suptitle("逐零后验对比：v1 旧管线（左列） vs v2 log-φ 先验（右列）"
                 "——同一批确定性种子 rep 0", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.text(0.5, 0.008, "红=oracle 结构零，绿=oracle 抽样零；黑刻度=该类的"
             "后验四分位点（25/50/75%）", ha="center", fontsize=9,
             color="#555555")
    out = os.path.join(BASE, "fig_a_zero_posterior_v1_vs_v2.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig_b(df, sel):
    metrics = [("pr_auc", "PR-AUC ↑"), ("brier", "Brier ↓"),
               ("phi_log_bias_abs", "|log(φ/φ真值)| ↓")]
    df = df.copy()
    df["phi_log_bias_abs"] = df["phi_log_bias"].abs()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    width = 0.35
    for ax, (m, lab) in zip(axes, metrics):
        xs = np.arange(len(CELL_ORDER))
        for k, (arm_key, color, name) in enumerate(
                [("v1", C_V1, "v1 旧管线"), ("v2", C_V2, "v2 最优λ")]):
            means, pts = [], []
            for cid in CELL_ORDER:
                if arm_key == "v1":
                    g = df[(df.cell_id == cid) & (df.arm == "v1")]
                else:
                    g = df[(df.cell_id == cid)
                           & (df.arm == f"v2_l{sel[cid]:g}")]
                means.append(g[m].mean())
                pts.append(g[m].to_numpy())
            xpos = xs + (k - 0.5) * width
            ax.bar(xpos, means, width * 0.85, color=color, alpha=0.85,
                   label=name)
            for x, p in zip(xpos, pts):
                ax.scatter([x] * len(p), p, color="#333333", s=9, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"格 {c}" for c in CELL_ORDER])
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("指标对比：v1 vs v2(逐 rep 均值, 黑点=逐 rep 值, reps 0-2)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(BASE, "fig_b_metric_bars.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def main():
    df = pd.read_csv(SUM_CSV)
    # 每格最优 λ（rep0 PR-AUC 主选、Brier 辅选）
    sel = {}
    for cid in CELL_ORDER:
        g = df[(df.cell_id == cid) & (df.arm != "v1") & (df.rep == 0)]
        g = g.sort_values(["pr_auc", "brier"], ascending=[False, True])
        sel[int(cid)] = float(g.iloc[0]["prior_lam"])
    print("lambda selection:", sel)
    fig_a(sel)
    fig_b(df, sel)


if __name__ == "__main__":
    main()
