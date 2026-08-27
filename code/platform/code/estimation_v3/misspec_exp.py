"""misspec_exp.py — 误设场景下 Godambe 三明治修正必要性的实验证据。

针对 README 已知局限 5（主校准 Exp 1 中 Godambe≈naive，修正必要性无证据），
构造两个"修正起效"的误设场景，并对比三档标准误：

  naive  : V = A⁻¹                （工作模型期望 Fisher 逆，假设逐 (i,j) 独立
                                    且信息矩阵等式成立）
  indep  : V = A⁻¹ B_ind A⁻¹      （逐观测 sandwich：B_ind = Σ_ij u_ij u_ijᵀ，
                                    修正边际方差误设，但忽略样本内跨类群相关）
  clust  : V = A⁻¹ B_cl A⁻¹       （逐样本聚类 sandwich：B_cl = Σ_i u_i u_iᵀ，
                                    即 composite_likelihood.godambe_covariance
                                    现有实现的 meat —— 检查结论：现有 meat 已
                                    按样本聚类，indep 档为本实验新增的对照档）

场景 A（因子耦合误设）：按完整模型的存在层因子结构生成，
  logit π_ij = α_j + u_jᵀλ_i,  λ_i ~ N(0, I_R),  u_jr ~ N(0, κ²)，
  D_ij | λ_i ~ Bern(π_ij · [1−g(N_i; θ̄_j, φ)])（给定 λ_i 逐类群独立）。
α_j 经 1D 求积校准使边际 π̄_j = E_λ[π_ij] 恰等于主校准目标 PI_CAL[j]，因此
**边际均值模型恰好正确**（D_ij 边际 ~ Bern(π̄_j[1−g])），唯一的误设是样本内
跨类群相关 —— 这是隔离"相关结构误设"的干净设计：θ̂、φ̂ 无伪真值偏差，
naive/indep 的失败可完全归因于忽略簇相关。严重度梯度 κ∈{0,0.5,1.0}：
κ=0 应复现 Exp 1 的 Godambe≈naive（边界 sanity check）。

场景 B（深度依赖检出效率误设）：
  D_ij | ρ_i ~ Bern(π_j · [1−g(ρ_i N_i; θ̄_j, φ)]),  ρ_i = exp(σ_ρ Z_i),
Z_i ~ N(0,1) iid（样本级随机检出效率，乘在有效深度上，跨类群共享）。边际
检出概率 π_j·E_ρ[1−g(ρN;θ,φ)] 不在工作模型族 {π(1−g(N;θ,φ))} 内，故
(a) θ̂、φ̂ 有伪真值偏差（sandwich 修正方差、不修正偏差——预期三档对真值
覆盖率都会随 σ_ρ 退化，这是"修正失效"的诚实边界）；(b) ρ_i 跨类群共享
再次引入样本内相关。为分离 (a)/(b)，每格用大 n 单次拟合估计伪真值
(π*,θ*,φ*)，同时报告"对真值"与"对伪真值"两套覆盖率。严重度
σ_ρ∈{0,0.3,0.6}。

每格 R=500 重复（默认；MC 标准误 ≈±1%），联合模式（φ 未知）拟合——只有
在共享 φ 的联合臂，跨类群相关才会通过 A 的耦合传播到全部参数的 SE
（profile 臂 A 跨类群块对角，indep 与 clust 的对角 SE 恒等，见 README）。

运行：python3 misspec_exp.py [--R 500] [--cores 2]
产出：results/misspec_factor_perrep.csv / _summary.csv
      results/misspec_depth_perrep.csv  / _summary.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, roots_hermite

import model
import composite_likelihood as cl

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# 与主校准 Exp 1 相同的设计常数（validate.py）
PHI_CAL = 3000.0
PI_CAL = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH_CAL = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4
N_SAMP = 200
Z95 = 1.959964

R_FACT = 2                              # 场景 A 因子数
KAPPA_GRID = (0.0, 0.75, 1.5)           # 场景 A 载荷尺度（严重度）
SIGMA_RHO_GRID = (0.0, 0.3, 0.6, 1.0)   # 场景 B log 效率噪声（严重度）

_EPS = 1e-12


def _depths(rng, n=N_SAMP, lo=1e3, hi=1e5):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(int)


# ---------------------------------------------------------------------------
# 三档协方差：naive / 逐观测 sandwich（indep）/ 逐样本聚类 sandwich（clust）
# ---------------------------------------------------------------------------

def indep_meat(psi, D, N):
    """逐观测 sandwich meat：B_ind = Σ_ij u_ij u_ijᵀ（联合模式参数序
    (γ, α_1..α_p, β_1..β_p)）。u_ij 仅支撑于 {γ, α_j, β_j}。
    与聚类 meat 的关系：B_cl = B_ind + Σ_i Σ_{j≠k} u_ij u_ikᵀ；类群专属
    参数（α_j, β_j）的对角元恒等，共享 γ 的对角不恒等（其逐样本得分本身
    是跨类群求和，平方后含交叉项）。"""
    n, p = D.shape
    k = 1 + 2 * p
    phi, pi, theta = cl._unpack(psi, p, None)
    q, dqg, dqa, dqb = cl._q_and_grad(N, theta, phi, pi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    r = (D - qc) / (qc * (1.0 - qc))          # Bernoulli 得分残差 (n,p)
    sg = r * dqg
    sa = r * dqa
    sb = r * dqb
    B = np.zeros((k, k))
    B[0, 0] = float((sg * sg).sum())
    for j in range(p):
        ia, ib = 1 + j, 1 + p + j
        B[0, ia] = B[ia, 0] = float((sg[:, j] * sa[:, j]).sum())
        B[0, ib] = B[ib, 0] = float((sg[:, j] * sb[:, j]).sum())
        B[ia, ia] = float((sa[:, j] * sa[:, j]).sum())
        B[ia, ib] = B[ib, ia] = float((sa[:, j] * sb[:, j]).sum())
        B[ib, ib] = float((sb[:, j] * sb[:, j]).sum())
    return B


def three_tier_covariance(psi, D, N, ridge=1e-10):
    """返回 (V_naive, V_indep, V_clust)。A 与岭正则同 godambe_covariance。"""
    A, U = cl._fisher_and_scores(psi, D, N, None)
    B_cl = U.T @ U
    B_in = indep_meat(psi, D, N)
    k = A.shape[0]
    A_reg = A + ridge * np.trace(A) / k * np.eye(k)
    A_inv = np.linalg.inv(A_reg)
    return A_inv, A_inv @ B_in @ A_inv, A_inv @ B_cl @ A_inv


def three_tier_se(fit, D, N):
    """由 fit_composite（联合模式）输出重建 ψ，返回三档估计尺度 SE：
    dict('gamma': (3,), 'alpha': (3,p), 'beta': (3,p))，行序 naive/indep/clust。"""
    p = D.shape[1]
    psi = cl._pack(fit["phi"], fit["pi"], fit["theta"], None)
    Vn, Vi, Vc = three_tier_covariance(psi, D, N)
    sd = [np.sqrt(np.maximum(np.diag(V), 0.0)) for V in (Vn, Vi, Vc)]
    return {
        "gamma": np.array([s[0] for s in sd]),
        "alpha": np.array([s[1:1 + p] for s in sd]),
        "beta": np.array([s[1 + p:1 + 2 * p] for s in sd]),
    }


# ---------------------------------------------------------------------------
# 场景 A：因子耦合误设生成器
# ---------------------------------------------------------------------------

def draw_loadings(kappa, p, R, seed=None):
    """逐严重度格固定一次的载荷矩阵 U (p,R)。

    结构：第 1 因子为**全正载荷共享因子**（u_j1 = κ·w_j，w_j ~ 半正态
    |N(0,1)|，一次性抽出后跨重复固定）——样本级"存在质量"随机效应，使所有
    类群对的 π_ij 正相关；若载荷以 0 为中心随机抽取，跨类群得分协方差
    符号相消、B_cl≈B_ind，误设效应被平均掉（试点实验证实）。其余 R−1 个
    因子为 0 均值随机载荷（κ=0 时 U≡0，退化为 Exp 1 的独立情形）。"""
    if seed is None:
        # 确定性种子（不能用内置 hash()：字符串哈希随进程随机化）
        seed = 40_000 + int(round(kappa * 100))
    rng = np.random.default_rng(seed)
    U = np.zeros((p, R))
    U[:, 0] = kappa * np.abs(rng.normal(size=p))
    if R > 1:
        U[:, 1:] = rng.normal(0.0, 0.3 * kappa, size=(p, R - 1))
    return U


def calibrate_alpha(pi0, s, n_quad=64):
    """求 α 使 E_Z[expit(α + sZ)] = π0（Z~N(0,1)，Gauss–Hermite + brentq）。

    场景 A 中 u_jᵀλ_i ~ N(0, s_j²)，s_j = κ‖u_j‖。校准后边际 π̄_j = π0，
    使边际均值模型恰好正确（纯相关结构误设）。"""
    if s == 0.0:
        return logit(pi0)
    x, w = roots_hermite(n_quad)
    z = np.sqrt(2.0) * x
    wn = w / np.sqrt(np.pi)

    def mean_pi(a):
        return float(np.dot(wn, expit(a + s * z))) - pi0

    return float(brentq(mean_pi, logit(pi0) - 6.0, logit(pi0) + 6.0,
                        xtol=1e-13, rtol=1e-13))


def calibrate_alphas(U, pi_cal=PI_CAL):
    """对载荷矩阵 U 的每个类群返回校准后的 α_j。"""
    s = np.linalg.norm(U, axis=1)
    return np.array([calibrate_alpha(pi_cal[j], s[j]) for j in range(len(pi_cal))])


def sim_factor(rng, U, alpha, N, theta=TH_CAL, phi=PHI_CAL):
    """场景 A：D_ij | λ_i ~ Bern(expit(α_j + u_jᵀλ_i) · [1−g(N_i;θ_j,φ)])。"""
    n = N.shape[0]
    lam = rng.normal(size=(n, U.shape[1]))
    pi_ij = expit(alpha[None, :] + lam @ U.T)          # (n,p)
    q = pi_ij * (1.0 - model.g_closed(N[:, None].astype(float),
                                      theta[None, :], phi))
    return (rng.random(q.shape) < q).astype(float)


# ---------------------------------------------------------------------------
# 场景 B：深度依赖检出效率误设生成器
# ---------------------------------------------------------------------------

def sim_depth(rng, sigma_rho, N, pi=PI_CAL, theta=TH_CAL, phi=PHI_CAL):
    """场景 B：D_ij | ρ_i ~ Bern(π_j · [1−g(ρ_i N_i;θ_j,φ)])，ρ_i=exp(σ_ρ Z)。"""
    n = N.shape[0]
    rho = np.exp(sigma_rho * rng.normal(size=n))
    q = pi[None, :] * (1.0 - model.g_closed((rho * N)[:, None].astype(float),
                                            theta[None, :], phi))
    return (rng.random(q.shape) < q).astype(float)


# ---------------------------------------------------------------------------
# 逐重复与汇总
# ---------------------------------------------------------------------------

def _offdiag_corr(D):
    """样本内跨类群检出相关的经验诊断：D 列间 Pearson 相关的非对角均值。"""
    C = np.corrcoef(D.T)
    p = C.shape[0]
    mask = ~np.eye(p, dtype=bool)
    return float(np.nanmean(C[mask]))


def _rep(job):
    """单个重复。job = (scenario, severity, seed)。返回 perrep 行 dict。"""
    scenario, severity, seed = job
    rng = np.random.default_rng(seed)
    p = len(PI_CAL)
    N = _depths(rng)
    if scenario == "factor":
        U = draw_loadings(severity, p, R_FACT)
        alpha = calibrate_alphas(U)
        D = sim_factor(rng, U, alpha, N)
    else:
        D = sim_depth(rng, severity, N)
    fit = cl.fit_composite(D, N)
    se = three_tier_se(fit, D, N)
    row = {"scenario": scenario, "severity": severity, "seed": seed,
           "success": bool(fit["success"]), "corr_offdiag": _offdiag_corr(D),
           "phi": fit["phi"],
           "sg_naive": se["gamma"][0], "sg_indep": se["gamma"][1],
           "sg_clust": se["gamma"][2]}
    for j in range(p):
        row[f"pi_{j}"] = fit["pi"][j]
        row[f"th_{j}"] = fit["theta"][j]
        row[f"sa_naive_{j}"] = se["alpha"][0, j]
        row[f"sa_indep_{j}"] = se["alpha"][1, j]
        row[f"sa_clust_{j}"] = se["alpha"][2, j]
        row[f"sb_naive_{j}"] = se["beta"][0, j]
        row[f"sb_indep_{j}"] = se["beta"][1, j]
        row[f"sb_clust_{j}"] = se["beta"][2, j]
    return row


def pseudo_truth(scenario, severity, n=60_000, seed=777):
    """场景误设下工作模型的伪真值 (π*, θ*, φ*)：大 n 单次拟合（n=6e4 时
    MC 噪声 ~0.4%，可忽略）。返回 dict。"""
    rng = np.random.default_rng(seed)
    p = len(PI_CAL)
    N = _depths(rng, n=n)
    if scenario == "factor":
        U = draw_loadings(severity, p, R_FACT)
        alpha = calibrate_alphas(U)
        D = sim_factor(rng, U, alpha, N)
    else:
        D = sim_depth(rng, severity, N)
    fit = cl.fit_composite(D, N)
    return {"pi": fit["pi"], "theta": fit["theta"], "phi": fit["phi"]}


def _cov(est, se, target):
    return float(((est - Z95 * se <= target) & (target <= est + Z95 * se)).mean())


def summarize(df, scenario, pseudo=None):
    """perrep → 每严重度格 × 每参数的偏差 / 经验 SD / 三档覆盖率 / SE 比。

    pseudo：若给定（场景 B），额外报告对伪真值的覆盖率与偏差，用于分离
    "sandwich 修正方差"（对伪真值覆盖率应恢复 0.95）与"偏差不可修正"
    （对真值覆盖率随严重度退化）。"""
    rows = []
    la_true, lb_true = logit(PI_CAL), np.log(TH_CAL)
    lg_true = np.log(PHI_CAL)
    for sev, sub in df.groupby("severity"):
        succ = float(sub["success"].mean())
        corr = float(sub["corr_offdiag"].mean())
        for j in range(len(PI_CAL)):
            ah = logit(sub[f"pi_{j}"].values)
            bh = np.log(sub[f"th_{j}"].values)
            sd_a, sd_b = ah.std(ddof=1), bh.std(ddof=1)
            row = {"scenario": scenario, "severity": sev, "param": f"taxon_{j}",
                   "success_rate": succ, "corr_offdiag": corr,
                   "logit_pi_bias": ah.mean() - la_true[j],
                   "log_theta_bias": bh.mean() - lb_true[j],
                   "sd_logit_pi": sd_a, "sd_log_theta": sd_b}
            for tier in ("naive", "indep", "clust"):
                sa, sb = sub[f"sa_{tier}_{j}"].values, sub[f"sb_{tier}_{j}"].values
                row[f"cov95_alpha_{tier}"] = _cov(ah, sa, la_true[j])
                row[f"cov95_beta_{tier}"] = _cov(bh, sb, lb_true[j])
                # SE 比用中位 SE：π̂/θ̂ 近边界时 logit/log 尺度 SE 有巨大离群值，
                # 均值被少数重复主导（试点：均值 20 vs 中位 ~0.5）
                row[f"seratio_alpha_{tier}"] = float(np.median(sa) / sd_a)
                row[f"seratio_beta_{tier}"] = float(np.median(sb) / sd_b)
            if pseudo is not None:
                la_p, lb_p = logit(pseudo["pi"][j]), np.log(pseudo["theta"][j])
                row["logit_pi_bias_pseudo"] = ah.mean() - la_p
                row["log_theta_bias_pseudo"] = bh.mean() - lb_p
                for tier in ("naive", "indep", "clust"):
                    sa, sb = (sub[f"sa_{tier}_{j}"].values,
                              sub[f"sb_{tier}_{j}"].values)
                    row[f"cov95_alpha_pseudo_{tier}"] = _cov(ah, sa, la_p)
                    row[f"cov95_beta_pseudo_{tier}"] = _cov(bh, sb, lb_p)
            rows.append(row)
        # φ 行（联合臂共享参数，跨类群相关的主要受害者）
        gh = np.log(sub["phi"].values)
        sd_g = gh.std(ddof=1)
        prow = {"scenario": scenario, "severity": sev, "param": "phi",
                "success_rate": succ, "corr_offdiag": corr,
                "log_phi_bias": gh.mean() - lg_true, "sd_log_phi": sd_g}
        for tier in ("naive", "indep", "clust"):
            sg = sub[f"sg_{tier}"].values
            prow[f"cov95_phi_{tier}"] = _cov(gh, sg, lg_true)
            prow[f"seratio_phi_{tier}"] = float(np.median(sg) / sd_g)
        if pseudo is not None:
            lg_p = np.log(pseudo["phi"])
            prow["log_phi_bias_pseudo"] = gh.mean() - lg_p
            for tier in ("naive", "indep", "clust"):
                sg = sub[f"sg_{tier}"].values
                prow[f"cov95_phi_pseudo_{tier}"] = _cov(gh, sg, lg_p)
        rows.append(prow)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_scenario(scenario, grid, R, cores, seed_base):
    from multiprocessing import Pool
    jobs = [(scenario, sev, seed_base + gi * 10_000 + r)
            for gi, sev in enumerate(grid) for r in range(R)]
    t0 = time.time()
    if cores > 1:
        with Pool(cores) as pool:
            rows = pool.map(_rep, jobs, chunksize=16)
    else:
        rows = [_rep(j) for j in jobs]
    df = pd.DataFrame(rows)
    perrep_path = OUT / f"misspec_{scenario}_perrep.csv"
    df.to_csv(perrep_path, index=False)
    # 伪真值（两场景都给；场景 A 因校准设计，伪真值应≈真值，作为校验）
    pseudo = {sev: pseudo_truth(scenario, sev) for sev in grid}
    summ = pd.concat([summarize(df[df["severity"] == sev], scenario,
                                pseudo=pseudo[sev])
                      for sev in grid])
    summ_path = OUT / f"misspec_{scenario}_summary.csv"
    summ.to_csv(summ_path, index=False)
    print("== 场景 %s（R=%d/格，%.0fs）==  写出 %s / %s"
          % (scenario, R, time.time() - t0, perrep_path.name, summ_path.name))
    for sev in grid:
        s = summ[summ["severity"] == sev]
        pr = s[s["param"] == "phi"].iloc[0]
        tx = s[s["param"] != "phi"]
        print("  严重度 %.1f：corr_offdiag=%.3f，cov95 φ (n/i/c) = "
              "%.3f/%.3f/%.3f；log-θ cov95 clust 范围 [%.3f, %.3f]，"
              "naive [%.3f, %.3f]"
              % (sev, pr["corr_offdiag"], pr["cov95_phi_naive"],
                 pr["cov95_phi_indep"], pr["cov95_phi_clust"],
                 tx["cov95_beta_clust"].min(), tx["cov95_beta_clust"].max(),
                 tx["cov95_beta_naive"].min(), tx["cov95_beta_naive"].max()))
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500, help="每格重复数（≥200）")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--scenario", choices=["factor", "depth", "both"],
                    default="both")
    args = ap.parse_args()
    if args.scenario in ("factor", "both"):
        run_scenario("factor", KAPPA_GRID, args.R, args.cores, seed_base=20_000)
    if args.scenario in ("depth", "both"):
        run_scenario("depth", SIGMA_RHO_GRID, args.R, args.cores, seed_base=60_000)


if __name__ == "__main__":
    main()
