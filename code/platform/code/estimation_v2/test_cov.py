"""test_cov.py — 里程碑 1（存在层协变量）单元测试。

覆盖：
  1. 解析梯度 vs 中心有限差分（φ 已知 / 联合两种模式）；
  2. φ 已知时逐类群分解目标 = 联合目标；
  3. Fisher 对称、A 的逐类群块结构、Godambe 协方差对称正定对角；
  4. 无协变量特例（K=1 截距）与 composite_likelihood 的 ℓ 一致；
  5. 模拟器经验检出率 vs 理论曲线（MC 容差）；
  6. 大样本单数据集 γ 恢复（宽松容差）。

运行：python3 -m pytest test_cov.py -v
"""

import numpy as np
import pytest
from scipy.special import expit, logit

import model
import model_cov
import composite_likelihood as cl
import composite_likelihood_cov as clc

RNG = np.random.default_rng(7)


def _make_problem(n=60, p=3, K=3, seed=1):
    rng = np.random.default_rng(seed)
    cols = [np.ones(n)]
    if K >= 2:
        cols.append(rng.integers(0, 2, n))
    if K >= 3:
        cols.append(rng.standard_normal(n))
    W = np.column_stack(cols)
    Gamma = rng.normal(0, 0.8, (p, K))
    Gamma[:, 0] = 1.5
    theta = np.array([8e-4, 1e-3, 1.2e-3])[:p]
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    phi = 3000.0
    # 直接按逐类群边际生成 D（测试目标函数即可，无需全三层）
    Pi = expit(W @ Gamma.T)
    q = Pi * (1.0 - model.g_closed(N[:, None], theta[None, :], phi))
    D = (rng.random((n, p)) < q).astype(float)
    return D, W, N, Gamma, theta, phi


def _fd_grad(fun, x, h=1e-5):
    g = np.zeros_like(x)
    for k in range(x.shape[0]):
        hh = h * max(1.0, abs(x[k]))
        xp, xm = x.copy(), x.copy()
        xp[k] += hh; xm[k] -= hh
        g[k] = (fun(xp) - fun(xm)) / (2 * hh)
    return g


# ---------------------------------------------------------------------------
# 1. 解析梯度 vs 有限差分
# ---------------------------------------------------------------------------

def test_grad_vs_fd_phi_known():
    D, W, N, G, theta, phi = _make_problem()
    psi = clc._pack(phi, G, theta, phi_known=phi)
    _, ga = clc._neg_loglik_grad(psi, D, W, N, phi)
    gn = _fd_grad(lambda x: clc.composite_loglik_cov(x, D, W, N, phi), psi)
    assert np.allclose(-ga, gn, rtol=1e-4, atol=1e-4)


def test_grad_vs_fd_joint():
    D, W, N, G, theta, phi = _make_problem()
    psi = clc._pack(phi, G, theta, phi_known=None)
    _, ga = clc._neg_loglik_grad(psi, D, W, N, None)
    gn = _fd_grad(lambda x: clc.composite_loglik_cov(x, D, W, N, None), psi)
    assert np.allclose(-ga, gn, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. φ 已知逐类群分解 = 联合目标
# ---------------------------------------------------------------------------

def test_phi_known_per_taxon_decomposition():
    D, W, N, G, theta, phi = _make_problem()
    full = clc.composite_loglik_cov(clc._pack(phi, G, theta, phi), D, W, N, phi)
    parts = sum(clc.composite_loglik_cov(
        clc._pack(phi, G[j:j + 1], theta[j:j + 1], phi),
        D[:, j:j + 1], W, N, phi) for j in range(D.shape[1]))
    assert abs(full - parts) < 1e-8


# ---------------------------------------------------------------------------
# 3. Fisher / Godambe 结构
# ---------------------------------------------------------------------------

def test_fisher_symmetric_block_structure():
    D, W, N, G, theta, phi = _make_problem()
    p, K = G.shape
    psi = clc._pack(phi, G, theta, phi)
    A, U = clc._fisher_and_scores(psi, D, W, N, phi)
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    # 跨类群块应为零（v_ij 支撑不相交）
    off, idx_G, idx_b = clc._layout(p, K, phi)
    par0 = np.concatenate([idx_G[0], [idx_b[0]]])
    par1 = np.concatenate([idx_G[1], [idx_b[1]]])
    assert np.allclose(A[np.ix_(par0, par1)], 0.0)
    # 类群内 γ–β 耦合应非零
    assert np.any(np.abs(A[np.ix_(idx_G[0], [idx_b[0]])]) > 0)
    Vg, Vn, _, _ = clc.godambe_covariance_cov(psi, D, W, N, phi)
    assert np.allclose(Vg, Vg.T)
    assert np.all(np.diag(Vg) > 0)


# ---------------------------------------------------------------------------
# 4. K=1（仅截距）特例退化为 composite_likelihood
# ---------------------------------------------------------------------------

def test_intercept_only_matches_baseline():
    D, W, N, G, theta, phi = _make_problem(K=1)
    p = D.shape[1]
    pi = expit(G[:, 0])
    ll_new = clc.composite_loglik_cov(clc._pack(phi, G, theta, phi),
                                      D, W, N, phi)
    psi_base = cl._pack(phi, pi, theta, phi)
    ll_base = cl.composite_loglik(psi_base, D, N, phi)
    assert abs(ll_new - ll_base) < 1e-10


# ---------------------------------------------------------------------------
# 5. 模拟器检出率 vs 理论曲线
# ---------------------------------------------------------------------------

def test_simulator_detection_rate():
    rng = np.random.default_rng(11)
    n, p, K = 4000, 3, 2
    W = np.column_stack([np.ones(n), rng.integers(0, 2, n)])
    Gamma = np.array([[1.4, 0.8], [1.7, -0.5], [2.0, 0.0]])
    theta_bar = np.array([8e-4, 1e-3, 1.2e-3])
    tb_full = np.concatenate([theta_bar, [1.0 - theta_bar.sum()]])
    G_full = np.vstack([Gamma, [20.0, 0.0]])   # bulk 类群 π≈1
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    phi = 3000.0
    Y, Z = model_cov.simulate_three_layer_cov(G_full, W, tb_full, phi, N, rng)
    D = (Y > 0).astype(float)
    Pi = expit(W @ Gamma.T)
    for j in range(p):
        theo = (Pi[:, j]
                * (1.0 - model.g_closed(N, theta_bar[j], phi))).mean()
        emp = D[:, j].mean()
        se = np.sqrt(theo * (1 - theo) / n)
        assert abs(emp - theo) < 4 * se + 3e-3   # E1 耦合余项留 3e-3


# ---------------------------------------------------------------------------
# 6. 大样本 γ 恢复（单数据集，宽松容差）
# ---------------------------------------------------------------------------

def test_gamma_recovery_large_sample():
    rng = np.random.default_rng(13)
    n, p, K = 4000, 4, 3
    W = np.column_stack([np.ones(n),
                         rng.integers(0, 2, n),
                         rng.standard_normal(n)])
    Gamma = np.array([[1.6, 0.8, 0.5], [1.8, -0.7, 0.0],
                      [2.0, 0.0, -0.6], [1.5, 0.5, 0.4]])
    theta_bar = np.array([8e-4, 9e-4, 1e-3, 1.1e-3])
    tb_full = np.concatenate([theta_bar, [1.0 - theta_bar.sum()]])
    G_full = np.vstack([Gamma, [20.0, 0.0, 0.0]])
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    phi = 3000.0
    Y, Z = model_cov.simulate_three_layer_cov(G_full, W, tb_full, phi, N, rng)
    D = (Y > 0).astype(float)[:, :p]
    fit = clc.fit_composite_cov(D, W, N, phi_known=phi)
    assert fit["success"]
    # 单数据集噪声级 |γ̂−γ| ≲ 2·SE（SE≈0.04–0.13）；取 0.25 宽松容差
    assert np.all(np.abs(fit["Gamma"] - Gamma) < 0.25)
    assert np.all(np.abs(np.log(fit["theta"]) - np.log(theta_bar)) < 0.3)


def test_wald_table_shape_and_null():
    D, W, N, G, theta, phi = _make_problem(n=200, p=3, K=3, seed=5)
    G[1, 1:] = 0.0
    fit = clc.fit_composite_cov(D, W, N, phi_known=phi)
    tab = clc.wald_table(fit)
    assert tab["z"].shape == G.shape
    assert np.all((tab["pval"] >= 0) & (tab["pval"] <= 1))
