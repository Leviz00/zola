"""test_misspec.py — misspec_exp（误设场景三档标准误实验）的单元测试。

覆盖：
  1. 逐观测 meat（indep）与逐样本聚类 meat（clust）的结构关系：
     对角恒等、B_cl − B_in = 跨类群交叉项、对称半定；
  2. α 校准（calibrate_alpha）：边际 E[expit(α+uᵀλ)] = π0（MC 复核）；
  3. 场景 A 生成器：校准后边际检出概率 = π̄_j[1−g]（解析对照）；
  4. 场景 B 生成器：σ_ρ=0 退化为基线；σ_ρ>0 时边际检出偏离 ρ≡1 曲线；
  5. 无端耦合（κ=0）时三档 SE 近似一致（复现 Exp 1 的 Godambe≈naive）；
  6. 强耦合（κ=1.5，正向共享因子）下修正起效：φ 的聚类 SE > naive，
     且 indep 与 naive 接近（边际均值模型正确 ⇒ 逐观测 sandwich 无可修正，
     只有聚类档能捕获跨类群相关）——三档分工的直接定量检验；
  7. 场景 B 误设下 θ̂/φ̂ 有伪真值偏差（sandwich 不修正偏差的边界演示）；
  8. 端到端小样本：_rep / summarize 输出结构与值域。

运行：python3 -m pytest test_misspec.py -v
"""

import numpy as np
import pytest
from scipy.special import expit, logit

import model
import composite_likelihood as cl
import misspec_exp as me

P = len(me.PI_CAL)


def _fit_and_se(D, N):
    fit = cl.fit_composite(D, N)
    assert fit["success"]
    return fit, me.three_tier_se(fit, D, N)


# ---------------------------------------------------------------------------
# 1. meat 结构关系
# ---------------------------------------------------------------------------

def test_indep_meat_diag_equals_cluster_diag():
    rng = np.random.default_rng(0)
    N = me._depths(rng)
    U = me.draw_loadings(1.0, P, me.R_FACT)
    D = me.sim_factor(rng, U, me.calibrate_alphas(U), N)
    fit = cl.fit_composite(D, N)
    psi = cl._pack(fit["phi"], fit["pi"], fit["theta"], None)
    B_in = me.indep_meat(psi, D, N)
    _, U_scores = cl._fisher_and_scores(psi, D, N, None)
    B_cl = U_scores.T @ U_scores
    # 类群专属参数（α_j, β_j）的对角恒等（聚类修正全部来自跨类群交叉项）；
    # 共享 γ 的对角不恒等：其逐样本得分本身即跨类群求和 Σ_j r_ij·dqg_ij，
    # 平方后含跨类群交叉项——这正是 γ 成为耦合主要受害者的原因
    assert np.allclose(np.diag(B_in)[1:], np.diag(B_cl)[1:], rtol=1e-10)
    # B_cl − B_in 恰为交叉项 Σ_i Σ_{j≠k} u_ij u_ikᵀ：类群专属对角为 0、对称
    X = B_cl - B_in
    assert np.allclose(np.diag(X)[1:], 0.0, atol=1e-8)
    assert np.allclose(X, X.T, atol=1e-8)
    assert X[0, 0] != 0.0
    # 两个 meat 均为对称半定
    for B in (B_in, B_cl):
        assert np.allclose(B, B.T, atol=1e-8)
        assert np.linalg.eigvalsh(B).min() > -1e-6


def test_indep_meat_matches_explicit_loop():
    """向量化 indep meat vs 逐观测显式循环（构造正确性）。"""
    rng = np.random.default_rng(1)
    N = me._depths(rng)[:20]
    D = me.sim_depth(rng, 0.3, N)
    fit = cl.fit_composite(D, N)
    psi = cl._pack(fit["phi"], fit["pi"], fit["theta"], None)
    B_fast = me.indep_meat(psi, D, N)
    n, p = D.shape
    k = 1 + 2 * p
    phi, pi, theta = cl._unpack(psi, p, None)
    q, dqg, dqa, dqb = cl._q_and_grad(N, theta, phi, pi)
    qc = np.clip(q, 1e-12, 1 - 1e-12)
    r = (D - qc) / (qc * (1 - qc))
    B_slow = np.zeros((k, k))
    for i in range(n):
        for j in range(p):
            u = np.zeros(k)
            u[0] = r[i, j] * dqg[i, j]
            u[1 + j] = r[i, j] * dqa[i, j]
            u[1 + p + j] = r[i, j] * dqb[i, j]
            B_slow += np.outer(u, u)
    assert np.allclose(B_fast, B_slow, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 2–3. α 校准与场景 A 边际
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pi0,s", [(0.85, 1.2), (0.95, 2.0), (0.5, 0.8)])
def test_calibrate_alpha(pi0, s):
    a = me.calibrate_alpha(pi0, s)
    rng = np.random.default_rng(2)
    z = rng.normal(size=2_000_000)
    emp = float(expit(a + s * z).mean())
    assert abs(emp - pi0) < 3e-3          # MC 标准误 ~3e-4


def test_factor_marginal_detection_matches_theory():
    """校准后边际 D_ij ~ Bern(π̄_j[1−g])：经验检出率 vs 解析曲线。"""
    rng = np.random.default_rng(3)
    U = me.draw_loadings(1.5, P, me.R_FACT)
    alpha = me.calibrate_alphas(U)
    N = np.full(20_000, 30_000)
    D = me.sim_factor(rng, U, alpha, N)
    emp = D.mean(axis=0)
    theo = me.PI_CAL * (1 - model.g_closed(30_000.0, me.TH_CAL, me.PHI_CAL))
    assert np.abs(emp - theo).max() < 4e-3   # MC 标准误 ~1e-3


def test_factor_within_sample_positive_correlation():
    """正向共享因子 ⇒ 残差 (D−q̄) 的样本内跨类群相关为正（误设存在）。"""
    rng = np.random.default_rng(4)
    U = me.draw_loadings(1.5, P, me.R_FACT)
    alpha = me.calibrate_alphas(U)
    N = me._depths(rng, n=4000)
    D = me.sim_factor(rng, U, alpha, N)
    resid = D - D.mean(axis=0, keepdims=True)
    C = np.corrcoef(resid.T)
    off = C[~np.eye(P, dtype=bool)]
    # 原始 D 列相关含共享 N_i 的混杂（两场景共有），这里只要求显著为正
    assert off.mean() > 0.05


# ---------------------------------------------------------------------------
# 4. 场景 B 生成器
# ---------------------------------------------------------------------------

def test_depth_sigma0_is_baseline():
    rng = np.random.default_rng(5)
    n = 20_000
    N = np.full(n, 30_000)
    D = me.sim_depth(rng, 0.0, N)
    emp = D.mean(axis=0)
    theo = me.PI_CAL * (1 - model.g_closed(30_000.0, me.TH_CAL, me.PHI_CAL))
    mc_se = np.sqrt(theo * (1 - theo) / n)
    assert (np.abs(emp - theo) < 4 * mc_se + 1e-4).all()


def test_depth_misspec_shifts_marginal():
    """σ_ρ>0 时 E_ρ[1−g(ρN)] ≠ 1−g(N)（Jensen），边际检出偏离 ρ≡1 曲线。"""
    rng = np.random.default_rng(6)
    N = np.full(20_000, 5_000)      # 未饱和深度，形状扭曲最敏感
    D = me.sim_depth(rng, 1.0, N)
    emp = D.mean(axis=0)
    theo = me.PI_CAL * (1 - model.g_closed(5_000.0, me.TH_CAL, me.PHI_CAL))
    assert np.abs(emp - theo).max() > 0.01


# ---------------------------------------------------------------------------
# 5–6. 三档 SE：无端耦合一致、强耦合修正起效
# ---------------------------------------------------------------------------

def _gamma_se_means(scenario, severity, R, seed0):
    sgs = np.array([[me._rep((scenario, severity, seed0 + r))[k]
                     for k in ("sg_naive", "sg_indep", "sg_clust")]
                    for r in range(R)])
    return sgs.mean(axis=0)


def test_no_coupling_three_tiers_agree():
    """κ=0（逐类群独立 + 正确模型）：三档 SE 中位相对差 <10%（Exp 1 复现）。"""
    m = _gamma_se_means("factor", 0.0, R=12, seed0=50_000)
    rel = (m - m[0]) / m[0]
    assert np.all(np.abs(rel) < 0.10)


def test_strong_coupling_correction_activates():
    """κ=1.5：聚类档 γ SE 显著大于 naive（>10%），indep≈naive（±5%）。

    这是"修正起效"的核心检验：边际均值模型正确（校准设计）⇒ 逐观测
    sandwich 无可修正；跨类群正相关只有逐样本聚类 meat 能捕获。"""
    m = _gamma_se_means("factor", 1.5, R=12, seed0=51_000)
    sn, si, sc = m
    assert sc / sn > 1.10
    assert abs(si / sn - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 7. 场景 B：伪真值偏差（sandwich 不修正偏差）
# ---------------------------------------------------------------------------

def test_depth_misspec_induces_bias():
    """σ_ρ=1.0：伪真值显著偏离真值（sandwich 不修正偏差的边界演示）。

    伪真值用大 n 单次拟合估计（n=3e4，MC 噪声 ~0.7%），φ* 应远低于
    3000（深度效率噪声把检出曲线压平，拟合以更小 φ 吸收），θ* 亦有
    可测偏移；同时 12 个 n=200 重复的均值应同向偏移。"""
    pt = me.pseudo_truth("depth", 1.0, n=30_000, seed=999)
    assert pt["phi"] < 0.7 * me.PHI_CAL
    assert np.abs(np.log(pt["theta"] / me.TH_CAL)).max() > 0.1
    rows = [me._rep(("depth", 1.0, 52_000 + r)) for r in range(12)]
    gh = np.log([r["phi"] for r in rows])
    assert np.mean(gh) < np.log(me.PHI_CAL) - 0.2


# ---------------------------------------------------------------------------
# 8. 端到端小样本
# ---------------------------------------------------------------------------

def test_end_to_end_small(tmp_path=None):
    rows = [me._rep(("factor", 0.75, 53_000 + r)) for r in range(6)]
    rows += [me._rep(("depth", 0.6, 54_000 + r)) for r in range(6)]
    import pandas as pd
    df = pd.DataFrame(rows)
    for scen in ("factor", "depth"):
        sub = df[df["scenario"] == scen]
        summ = me.summarize(sub, scen)
        assert set(summ["param"]) == {f"taxon_{j}" for j in range(P)} | {"phi"}
        for c in summ.columns:
            if c.startswith("cov95_"):
                assert summ[c].dropna().between(0, 1).all()
        assert summ["success_rate"].between(0, 1).all()
