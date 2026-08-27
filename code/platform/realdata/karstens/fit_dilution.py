"""fit_dilution.py — Karstens2019 稀释系列：检出曲线形状验证（任务 A 第 2 步）。

设计（n=9 mock 样本，1 样本/稀释度，属级）：
  - 核心 mock 类群 = D0（neat）中检出的 17 个属；
  - 已知稀释设计给出 θ 的**已知相对变化**：θ_j(k) = θ_j·d_k，d_k=3^{-k}
    （名义梯度；另用经验梯子 d_emp 与 DNA 浓度比作敏感性）；
  - θ_j 锚定自 D0 计数幅度：θ̂_j = Y_D0,j / N_D0（计数侧信息，避免 π–θ 脊）；
  - 模型 q_jk = π_j·[1 − g(N_k; θ_j·d_k, φ)]，共享 φ + 逐类群 π_j，
    Bernoulli 复合似然（沿用 code/estimation 的 g 闭式）。

形状检验三层：
  (S) 共享 φ 模型（18 参数）：φ 由梯子曲率绝对刻度识别（θ 已锚定）；
  (P) 乘积泛函模型（34 参数）：q=π_j(1−exp(−a_j·d_k))，a_j 逐类群自由——
      若模型正确，â_j ∝ θ_j（比例 = φS_N(φ)），检验 â_j/θ_j 的离散度；
  (F) 自由逐层模型（26 参数）：q_jk=π_j·h_k，h_k 逐稀释度自由——捕获任意
      随稀释单调下降的共有形状，作为模型 (S) 失拟的对照。
GOF：逐稀释度观测检出数 vs Σ_j q̂_jk（二项 SE），loglik 对比，φ 剖面似然。

输出：
  results/dilution_empirical.csv      逐稀释度经验统计（检出数、经验梯子等）
  results/dilution_core_taxa.csv      核心类群锚定 θ、逐层检出指示、预测概率
  results/dilution_fit_models.csv     三模型 loglik/参数数/φ̂ 与剖面 CI
  results/dilution_phi_profile.csv    φ 剖面对数似然（名义梯子）
  results/dilution_gof_by_level.csv   逐稀释度观测 vs 预测检出数
  results/dilution_product_test.csv   模型 P 的 â_j vs θ̂_j（形状一致性检验）
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import log_g  # noqa: E402

OUT = Path("/mnt/agents/output/realdata/karstens")
RES = OUT / "results"

GAMMA_BOUNDS = (np.log(0.05), np.log(1e6))
ALPHA_BOUNDS = (logit(1e-4), logit(0.9999))


def load():
    z = np.load(OUT / "data_karstens_genus.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 模型 S：共享 φ，θ 锚定，q_jk = π_j (1−g(N_k; θ_j d_k, φ))
# ---------------------------------------------------------------------------

def fit_shared_phi(D, N, theta_anchor, d, gamma0_grid=(2.0, 4.0, 6.0, 8.0)):
    """D: (K,J) 检出指示；N: (K,)；theta_anchor: (J,)；d: (K,) 稀释倍数。"""
    K, J = D.shape
    th = theta_anchor[None, :] * d[:, None]          # (K,J)
    th = np.clip(th, 1e-12, 1.0 - 1e-12)
    Nc = N[:, None].astype(float)

    def nll_grad(psi):
        gamma = psi[0]
        alpha = psi[1:]
        phi = np.exp(gamma)
        pi = expit(alpha)
        g = np.exp(log_g(Nc, th, phi))
        q = np.clip(pi[None, :] * (1.0 - g), 1e-12, 1 - 1e-12)
        ll = (D * np.log(q) + (1 - D) * np.log1p(-q)).sum()
        r = (D - q) / (q * (1 - q))                    # (K,J)
        # ∂q/∂γ = -π g ∂logg/∂φ · φ
        a = phi * (1.0 - th)
        from scipy.special import digamma
        dlogg_dphi = ((1 - th) * (digamma(a + Nc) - digamma(a))
                      - (digamma(phi + Nc) - digamma(phi)))
        dq_dgamma = -pi[None, :] * g * dlogg_dphi * phi
        s_gamma = float((r * dq_dgamma).sum())
        dq_dalpha = (1.0 - g) * (pi * (1 - pi))[None, :]
        s_alpha = (r * dq_dalpha).sum(axis=0)
        return -ll, -np.concatenate([[s_gamma], s_alpha])

    best = None
    for g0 in gamma0_grid:
        s = np.concatenate([[g0], np.full(J, logit(0.9))])
        res = minimize(nll_grad, s, method="L-BFGS-B", jac=True,
                       bounds=[GAMMA_BOUNDS] + [ALPHA_BOUNDS] * J,
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    gamma = best.x[0]
    pi = expit(best.x[1:])
    phi = np.exp(gamma)
    g = np.exp(log_g(Nc, th, phi))
    q = pi[None, :] * (1.0 - g)
    return {"phi": phi, "pi": pi, "q": q, "loglik": -best.fun,
            "success": bool(best.success) or np.max(np.abs(best.jac)) < 1e-3,
            "k_params": 1 + J}


def profile_phi(D, N, theta_anchor, d, grid):
    """φ 剖面：固定 φ，逐类群 1 维优化 logit π_j（快速）。"""
    K, J = D.shape
    th = np.clip(theta_anchor[None, :] * d[:, None], 1e-12, 1 - 1e-12)
    Nc = N[:, None].astype(float)
    out = []
    for phi in grid:
        g = np.exp(log_g(Nc, th, phi))                 # (K,J)
        ll = 0.0
        for j in range(J):
            amp = 1.0 - g[:, j]
            # 1 维：q_k = π·amp_k；π̂ = ΣD·? 无闭式（amp 非常数），用标量优化
            def f(a, amp=amp, d=D[:, j]):
                pi = expit(a)
                q = np.clip(pi * amp, 1e-12, 1 - 1e-12)
                return -(d * np.log(q) + (1 - d) * np.log1p(-q)).sum()
            r = minimize(f, [logit(0.9)], method="L-BFGS-B",
                         bounds=[ALPHA_BOUNDS])
            ll += -r.fun
        out.append(ll)
    return np.array(out)


# ---------------------------------------------------------------------------
# 模型 P：乘积泛函 q = π_j (1 − exp(−a_j d_k))，逐类群 (π_j, a_j) 独立
# ---------------------------------------------------------------------------

def fit_product(D, d):
    K, J = D.shape
    pis, aa, ll = [], [], 0.0
    dc = d[:, None].astype(float)
    for j in range(J):
        dj = D[:, j:j + 1]

        def nll(psi, dj=dj):
            a_, b_ = psi
            pi, aa_ = expit(a_), np.exp(b_)
            q = np.clip(pi * (1.0 - np.exp(-aa_ * dc)), 1e-12, 1 - 1e-12)
            return -(dj * np.log(q) + (1 - dj) * np.log1p(-q)).sum()

        best = None
        for a0 in (logit(0.5), logit(0.95)):
            for b0 in (np.log(0.1), np.log(3.0), np.log(100.0)):
                r = minimize(nll, [a0, b0], method="L-BFGS-B",
                             bounds=[ALPHA_BOUNDS, (np.log(1e-6), np.log(1e8))])
                if best is None or r.fun < best.fun:
                    best = r
        pis.append(expit(best.x[0]))
        aa.append(np.exp(best.x[1]))
        ll += -best.fun
    return {"pi": np.array(pis), "a": np.array(aa), "loglik": ll,
            "k_params": 2 * J}


# ---------------------------------------------------------------------------
# 模型 F：q_jk = π_j · h_k（h_k 逐稀释度自由，共享形状）
# ---------------------------------------------------------------------------

def fit_free_level(D):
    K, J = D.shape
    # h_K 相对 D0 归一：h_0 = 1（可吸收进 π），h_1..h_{K-1} ∈ (0,1) 自由
    def nll(psi):
        alpha = psi[:J]
        eta = psi[J:]                       # K-1 个 logit h
        pi = expit(alpha)
        h = np.concatenate([[1.0], expit(eta)])
        q = np.clip(pi[None, :] * h[:, None], 1e-12, 1 - 1e-12)
        return -(D * np.log(q) + (1 - D) * np.log1p(-q)).sum()

    best = None
    for a0 in (logit(0.8), logit(0.99)):
        s = np.concatenate([np.full(J, a0), np.full(K - 1, logit(0.5))])
        r = minimize(nll, s, method="L-BFGS-B",
                     bounds=[ALPHA_BOUNDS] * J + [ALPHA_BOUNDS] * (K - 1),
                     options={"maxiter": 3000, "ftol": 1e-14})
        if best is None or r.fun < best.fun:
            best = r
    pi = expit(best.x[:J])
    h = np.concatenate([[1.0], expit(best.x[J:])])
    return {"pi": pi, "h": h, "loglik": -best.fun, "k_params": J + K - 1}


# ---------------------------------------------------------------------------

def main():
    z = load()
    Y, depths = z["Y"].astype(float), z["depths"].astype(float)
    taxa = z["taxa"].astype(str)
    is_blank = z["is_blank"].astype(bool)
    dil = z["dil_factor"].astype(float)
    dna = z["dna_conc"].astype(float)
    samples = z["samples"].astype(str)

    mock = ~is_blank
    Ym, Nm = Y[mock], depths[mock]
    dm, dnam = dil[mock], dna[mock]
    sm = samples[mock]
    K = int(mock.sum())
    D = (Ym > 0).astype(float)

    # 核心类群：D0 检出
    core_mask = Ym[0] > 0
    J = int(core_mask.sum())
    core_taxa = taxa[core_mask]
    Dc = D[:, core_mask]
    theta_anchor = Ym[0, core_mask] / Nm[0]

    # ---- 经验统计 ---------------------------------------------------------
    rows = []
    core_reads = Ym[:, core_mask].sum(axis=1)
    d_emp = (core_reads / Nm) / (core_reads[0] / Nm[0])
    d_dna = dnam / dnam[0]
    for k in range(K):
        rows.append({
            "sample": sm[k], "level": k, "depth": int(Nm[k]),
            "nominal_d": dm[k], "empirical_d_core_reads": d_emp[k],
            "dna_conc_d": d_dna[k],
            "core_detected": int(Dc[k].sum()), "core_total": J,
            "noncore_detected": int(D[k, ~core_mask].sum()),
            "noncore_total": int((~core_mask).sum()),
        })
    df_emp = pd.DataFrame(rows)
    df_emp.to_csv(RES / "dilution_empirical.csv", index=False)
    print(df_emp.to_string(index=False))

    # 单调性：核心类群检出数 vs 稀释度
    from scipy.stats import spearmanr
    sp_core = spearmanr(df_emp["level"], df_emp["core_detected"])
    sp_noncore = spearmanr(df_emp["level"], df_emp["noncore_detected"])
    print(f"Spearman(level, core_detected)   = {sp_core.statistic:.3f} "
          f"(p={sp_core.pvalue:.3g})")
    print(f"Spearman(level, noncore_detected)= {sp_noncore.statistic:.3f} "
          f"(p={sp_noncore.pvalue:.3g})")

    # ---- 三模型拟合（名义梯子）-------------------------------------------
    fitS = fit_shared_phi(Dc, Nm, theta_anchor, dm)
    fitS_emp = fit_shared_phi(Dc, Nm, theta_anchor, d_emp)
    fitS_dna = fit_shared_phi(Dc, Nm, theta_anchor, d_dna)
    fitS_flat = fit_shared_phi(Dc, Nm, theta_anchor, np.ones(K))  # 无稀释零模型
    fitP = fit_product(Dc, d_emp)
    fitF = fit_free_level(Dc)

    # D1 脱落类群（非单调异常诊断）
    dropped_D1 = core_taxa[(Dc[1] == 0)]
    print(f"[anomaly] taxa NOT detected at D1 ({len(dropped_D1)}):",
          list(dropped_D1))

    # φ 剖面（经验梯子——主分析）
    grid = np.logspace(np.log10(0.5), 6, 48)
    ll_prof = profile_phi(Dc, Nm, theta_anchor, d_emp, grid)
    pd.DataFrame({"phi": grid, "profile_loglik": ll_prof}).to_csv(
        RES / "dilution_phi_profile.csv", index=False)
    ll_max = ll_prof.max()
    ci = grid[ll_prof >= ll_max - 1.92]     # 95% 剖面 CI
    phi_lo, phi_hi = (ci[0], ci[-1]) if len(ci) else (np.nan, np.nan)

    models = pd.DataFrame([
        {"model": "S_shared_phi_nominal", "loglik": fitS["loglik"],
         "k_params": fitS["k_params"], "phi_hat": fitS["phi"],
         "phi_ci_lo": np.nan, "phi_ci_hi": np.nan, "success": fitS["success"]},
        {"model": "S_shared_phi_empirical_ladder", "loglik": fitS_emp["loglik"],
         "k_params": fitS_emp["k_params"], "phi_hat": fitS_emp["phi"],
         "phi_ci_lo": phi_lo, "phi_ci_hi": phi_hi, "success": fitS_emp["success"]},
        {"model": "S_shared_phi_dna_conc_ladder", "loglik": fitS_dna["loglik"],
         "k_params": fitS_dna["k_params"], "phi_hat": fitS_dna["phi"],
         "phi_ci_lo": np.nan, "phi_ci_hi": np.nan, "success": fitS_dna["success"]},
        {"model": "S_flat_no_dilution", "loglik": fitS_flat["loglik"],
         "k_params": fitS_flat["k_params"], "phi_hat": fitS_flat["phi"],
         "phi_ci_lo": np.nan, "phi_ci_hi": np.nan, "success": fitS_flat["success"]},
        {"model": "P_product_functional_emp_ladder", "loglik": fitP["loglik"],
         "k_params": fitP["k_params"], "phi_hat": np.nan,
         "phi_ci_lo": np.nan, "phi_ci_hi": np.nan, "success": True},
        {"model": "F_free_per_level", "loglik": fitF["loglik"],
         "k_params": fitF["k_params"], "phi_hat": np.nan,
         "phi_ci_lo": np.nan, "phi_ci_hi": np.nan, "success": True},
    ])
    models["delta_loglik_vs_S_emp"] = models["loglik"] - fitS_emp["loglik"]
    models.to_csv(RES / "dilution_fit_models.csv", index=False)
    print(models.to_string(index=False))

    # ---- 逐类群明细与 GOF --------------------------------------------------
    hF = fitF["h"]                       # 已含 h_0=1 归一
    det_pattern = ["".join(str(int(x)) for x in Dc[:, j]) for j in range(J)]
    df_taxa = pd.DataFrame({
        "taxon": core_taxa, "theta_anchor_D0": theta_anchor,
        "pi_hat_S_emp": fitS_emp["pi"], "pi_hat_P": fitP["pi"],
        "a_hat_P": fitP["a"],
        "detection_pattern_D0_to_D8": det_pattern,
        "n_detected_of_9": Dc.sum(axis=0).astype(int),
    })
    df_taxa["a_over_theta"] = df_taxa["a_hat_P"] / df_taxa["theta_anchor_D0"]
    for k in range(K):
        df_taxa[f"q_hat_S_emp_D{k}"] = fitS_emp["q"][k]
    df_taxa.to_csv(RES / "dilution_core_taxa.csv", index=False)

    gof = pd.DataFrame({
        "level": np.arange(K), "sample": sm,
        "observed_detected": Dc.sum(axis=1).astype(int),
        "expected_S_nominal": fitS["q"].sum(axis=1),
        "expected_S_empirical": fitS_emp["q"].sum(axis=1),
        "expected_S_flat": fitS_flat["q"].sum(axis=1),
        "expected_F_free": (fitF["pi"][None, :] * hF[:, None]).sum(axis=1),
        "binom_sd_S_empirical": np.sqrt(
            (fitS_emp["q"] * (1 - fitS_emp["q"])).sum(axis=1)),
    })
    gof["residual_S_empirical"] = (
        (gof["observed_detected"] - gof["expected_S_empirical"])
        / gof["binom_sd_S_empirical"])
    gof.to_csv(RES / "dilution_gof_by_level.csv", index=False)
    print(gof.to_string(index=False))

    # 模型 P 形状一致性：log â_j vs log θ̂_j 应近似斜率 1
    # （多数核心类群检出饱和 → â_j 巨大不具信息，仅报告未饱和子集）
    mask = (df_taxa["a_hat_P"] > 0) & np.isfinite(df_taxa["a_over_theta"]) \
        & (df_taxa["a_hat_P"] < 1e4)
    n_saturated = int((df_taxa["a_hat_P"] >= 1e4).sum())
    print(f"[P-test] {n_saturated}/{J} taxa saturated (a_hat>=1e4, "
          f"detection ~ certain at all levels; uninformative for shape)")
    la = np.log(df_taxa.loc[mask, "a_hat_P"])
    lt = np.log(df_taxa.loc[mask, "theta_anchor_D0"])
    slope, intercept = np.polyfit(lt, la, 1)
    resid = la - (slope * lt + intercept)
    prod_test = df_taxa.loc[mask, ["taxon", "theta_anchor_D0", "a_hat_P",
                                   "a_over_theta"]].copy()
    prod_test["log_a_resid_vs_slope1"] = la.values - (lt.values + np.median(la - lt))
    prod_test.to_csv(RES / "dilution_product_test.csv", index=False)
    print(f"[P-test] slope(log a ~ log theta) = {slope:.3f} (model predicts 1); "
          f"sd(log a/theta) = {float(np.std(la - lt)):.3f} dex-free")


if __name__ == "__main__":
    main()
