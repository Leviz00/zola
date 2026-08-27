"""blank_anchor.py — 空白锚定 ρ 的首次演示（任务 A 第 3 步；命题 (iv) π–ρ 混杂）。

设计（概念验证规模，n=9 mock 样本 + 1 空白，属级 199 属）：
  - 空白谱：c_j = Y_blank,j / N_blank（Urine Extraction Blank，深度 189,779，
    149/199 属检出——空白本身即丰富污染谱）。
  - 三模型（同一复合似然框架、逐类群独立 MLE、共享 φ 剖面网格）：
      U  未锚定（ρ≡1）:
           q = π_j·[1 − g(N; θ_j, φ)]                                   (π,θ)
      A1 确定性 δ 锚定（朴素方案，预期被否定）:
           q = 1 − (1 − π_j[1−g(N;θ_j,φ)])(1 − δ_j(N)),
           δ_j(N) = 1−(1−c_j)^N                                          (π,θ)
      A2 随机 ρ 锚定（空白锚定每次读数污染效率 c_j，数据决定污染发生率 ρ_j）:
           q = 1 − (1 − π_j[1−g(N;θ_j,φ)])(1 − ρ_j·h_j(N)),
           h_j(N) = 1−(1−c_j)^N                                          (π,θ,ρ)
    A1 假设"污染在每一份样本中以空白丰度系统性存在"；A2 只假设"污染若发生，
    其每次读数效率由空白谱锚定"，发生率 ρ_j 逐类群自由（交叉污染是零星的）。
    c_j=0（空白未检出）时 A1/A2 退化为 U（ρ_j 固定 0）。
  - 演示量：π̂ 锚定前后变化（U vs A2） vs 空白谱强度；loglik 对比；
    φ̂ 三模型对比。

诚实性声明：单空白样本 → c_j 为点估计；空白与 mock 同批次假设未验证；
π_j 与 ρ_j 在单批数据内仅有深度变化提供的弱分离（命题 (iv) 的脊在本演示
中可观测）。结论为概念验证。

输出：
  results/blank_spectrum.csv        空白谱 c_j、h_j（中位深度）、检出统计
  results/anchor_pertaxon.csv       逐属三模型 (π̂, θ̂, ρ̂, loglik) 对比
  results/anchor_summary.csv        φ̂、总 loglik、π̂ 汇总、Spearman 相关
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import log_g  # noqa: E402
from composite_likelihood import _fit_single_taxon_phi_known  # noqa: E402

OUT = Path("/mnt/agents/output/realdata/karstens")
RES = OUT / "results"

PI_B = (logit(1e-4), logit(0.9999))
TH_B = (np.log(1e-8), np.log(0.9))


def _bern_ll(d, q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return float((d * np.log(q) + (1 - d) * np.log1p(-q)).sum())


def fit_taxon_A1(d, N, phi, delta):
    """A1：确定性 δ 锚定，逐属 (logit π, log θ) MLE。"""
    def nll(psi):
        pi, th = expit(psi[0]), np.exp(psi[1])
        g = np.exp(log_g(N, th, phi))
        return -_bern_ll(d, 1.0 - (1.0 - pi * (1.0 - g)) * (1.0 - delta))

    best = None
    for t0 in (1e-6, 1e-4, 1e-2):
        for p0 in (logit(0.1), logit(0.9)):
            r = minimize(nll, [p0, np.log(t0)], method="L-BFGS-B",
                         bounds=[PI_B, TH_B])
            if best is None or r.fun < best.fun:
                best = r
    return expit(best.x[0]), np.exp(best.x[1]), -best.fun


def fit_taxon_A2(d, N, phi, h):
    """A2：随机 ρ 锚定，逐属 (logit π, log θ, logit ρ) MLE。

    h_j(N) = 1−(1−c_j)^N 固定；c_j=0 时调用方应直接用 U 模型。
    """
    def nll(psi):
        pi, th, rho = expit(psi[0]), np.exp(psi[1]), expit(psi[2])
        g = np.exp(log_g(N, th, phi))
        return -_bern_ll(d, 1.0 - (1.0 - pi * (1.0 - g)) * (1.0 - rho * h))

    best = None
    for t0 in (1e-6, 1e-4, 1e-2):
        for r0 in (logit(0.05), logit(0.5), logit(0.95)):
            r = minimize(nll, [logit(0.5), np.log(t0), r0],
                         method="L-BFGS-B", bounds=[PI_B, TH_B, PI_B])
            if best is None or r.fun < best.fun:
                best = r
    return expit(best.x[0]), np.exp(best.x[1]), expit(best.x[2]), -best.fun


def main():
    z = np.load(OUT / "data_karstens_genus.npz", allow_pickle=True)
    Y, depths = z["Y"].astype(float), z["depths"].astype(float)
    taxa = z["taxa"].astype(str)
    is_blank = z["is_blank"].astype(bool)

    Ym, Nm = Y[~is_blank], depths[~is_blank].astype(float)
    Yb, Nb = Y[is_blank][0], depths[is_blank][0]
    D = (Ym > 0).astype(float)
    n, p = D.shape

    c = Yb / Nb                                       # 空白谱
    h = 1.0 - (1.0 - c)[None, :] ** Nm[:, None]       # (n, p) 污染效率项
    h = np.clip(h, 0.0, 1 - 1e-12)
    N_med = float(np.median(Nm))
    h_med = 1.0 - (1.0 - c) ** N_med

    # ---- φ 剖面（粗网格，三模型共享）--------------------------------------
    grid = np.logspace(0, 6, 16)
    tot = {"U": np.zeros_like(grid), "A1": np.zeros_like(grid),
           "A2": np.zeros_like(grid)}
    for gi, phi in enumerate(grid):
        for j in range(p):
            r = _fit_single_taxon_phi_known(D[:, j], Nm, phi)
            tot["U"][gi] += -r.fun
            tot["A1"][gi] += fit_taxon_A1(D[:, j], Nm, phi, h[:, j])[2]
            if c[j] > 0:
                tot["A2"][gi] += fit_taxon_A2(D[:, j], Nm, phi, h[:, j])[3]
            else:
                tot["A2"][gi] += -r.fun
        print(f"  phi={phi:.3g} U={tot['U'][gi]:.2f} A1={tot['A1'][gi]:.2f} "
              f"A2={tot['A2'][gi]:.2f}", flush=True)
    phi_best = {m: grid[int(np.argmax(tot[m]))] for m in tot}
    print("phi_hat:", phi_best)

    # ---- 最优 φ 处逐属终拟合 ----------------------------------------------
    rows = []
    for j in range(p):
        r = _fit_single_taxon_phi_known(D[:, j], Nm, phi_best["U"])
        pi_u, th_u, ll_u = expit(r.x[0]), np.exp(r.x[1]), -r.fun
        pi_1, th_1, ll_1 = fit_taxon_A1(D[:, j], Nm, phi_best["A1"], h[:, j])
        if c[j] > 0:
            pi_2, th_2, rho_2, ll_2 = fit_taxon_A2(D[:, j], Nm,
                                                   phi_best["A2"], h[:, j])
        else:
            pi_2, th_2, rho_2, ll_2 = pi_u, th_u, 0.0, ll_u
        rows.append({
            "taxon": taxa[j], "prevalence_mock": D[:, j].mean(),
            "blank_rel_abund": c[j], "blank_reads": int(Yb[j]),
            "h_at_median_depth": h_med[j],
            "pi_hat_U": pi_u, "theta_hat_U": th_u, "loglik_U": ll_u,
            "pi_hat_A1": pi_1, "theta_hat_A1": th_1, "loglik_A1": ll_1,
            "pi_hat_A2": pi_2, "theta_hat_A2": th_2, "rho_hat_A2": rho_2,
            "loglik_A2": ll_2,
        })
    df = pd.DataFrame(rows)
    df["pi_shrinkage_A2"] = df["pi_hat_U"] - df["pi_hat_A2"]
    df["delta_loglik_A2_vs_U"] = df["loglik_A2"] - df["loglik_U"]
    df.to_csv(RES / "anchor_pertaxon.csv", index=False)

    df[["taxon", "blank_rel_abund", "blank_reads", "h_at_median_depth",
        "prevalence_mock"]].sort_values(
        "blank_rel_abund", ascending=False).to_csv(RES / "blank_spectrum.csv",
                                                   index=False)

    inb = df["blank_reads"] > 0
    sp = spearmanr(df["blank_rel_abund"], df["prevalence_mock"])
    ll_tot = {m: float(df[f"loglik_{m}"].sum()) for m in ("U", "A1", "A2")}
    summ = pd.DataFrame([{
        "n_samples": n, "p_genera": p,
        "blank_depth": int(Nb), "n_genera_in_blank": int(inb.sum()),
        "phi_hat_U": phi_best["U"], "phi_hat_A1": phi_best["A1"],
        "phi_hat_A2": phi_best["A2"],
        **{f"loglik_{m}": ll_tot[m] for m in ll_tot},
        "delta_loglik_A1_vs_U": ll_tot["A1"] - ll_tot["U"],
        "delta_loglik_A2_vs_U": ll_tot["A2"] - ll_tot["U"],
        "spearman_blankabund_vs_mockprevalence": sp.statistic,
        "spearman_p": sp.pvalue,
        "n_pi_ge_half_U": int((df["pi_hat_U"] >= 0.5).sum()),
        "n_pi_ge_half_A2": int((df["pi_hat_A2"] >= 0.5).sum()),
        "median_pi_U_inblank": float(df.loc[inb, "pi_hat_U"].median()),
        "median_pi_A2_inblank": float(df.loc[inb, "pi_hat_A2"].median()),
        "median_pi_U_notinblank": float(df.loc[~inb, "pi_hat_U"].median()),
        "median_pi_A2_notinblank": float(df.loc[~inb, "pi_hat_A2"].median()),
        "median_shrinkage_inblank": float(df.loc[inb, "pi_shrinkage_A2"].median()),
        "median_shrinkage_notinblank":
            float(df.loc[~inb, "pi_shrinkage_A2"].median()),
        "max_shrinkage": float(df["pi_shrinkage_A2"].max()),
        "n_genera_A2_better_2ll_gt_3.84":
            int((df["delta_loglik_A2_vs_U"] * 2 > 3.84).sum()),
    }])
    summ.to_csv(RES / "anchor_summary.csv", index=False)
    print(summ.T.to_string())
    top = df.sort_values("pi_shrinkage_A2", ascending=False).head(15)
    print(top[["taxon", "prevalence_mock", "blank_rel_abund",
               "pi_hat_U", "pi_hat_A2", "rho_hat_A2",
               "pi_shrinkage_A2", "delta_loglik_A2_vs_U"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    main()
