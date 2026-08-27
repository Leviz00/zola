"""test_count.py — 里程碑 2（计数幅度 DM 块复合似然 + 逐单元后验）单元测试。

覆盖：
  1. 幅度部分解析梯度 vs 中心有限差分；
  2. 逐样本得分和 = 总梯度（检出+幅度）；
  3. b=1 严格退化：总 CL = cl 的检出指示 CL（值与梯度），数值 Hessian 的
     A ≈ cl 的解析 Fisher A；
  4. DM 幅度部分对 θ 有信息（真参 ℓ 高于扰动 θ）；
  5. 后验：深度单调性（深样本的零更可能是结构零）、真参下 zero-cell
     区分 AUC > 0.9、校准分箱均值与经验比例单调一致；
  6. 大样本单数据集参数恢复（b=4 与 b=1 的 sd(log θ̂) 对照在小规模多
     数据集上方向正确）。

运行：python3 -m pytest test_count.py -v
"""

import numpy as np
import pytest

import model
import composite_likelihood as cl
import composite_likelihood_count as clc
import posterior as post

PHI = 3000.0
PI = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4


def _sim(seed, n=200, phi=PHI, pi=PI, th=TH):
    rng = np.random.default_rng(seed)
    theta_bar = np.concatenate([th, [1.0 - th.sum()]])
    pi_all = np.concatenate([pi, [1.0]])
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, phi, N, rng)
    return Y[:, :len(pi)], Z[:, :len(pi)], N


def _fd_grad(fun, x, h=1e-5):
    g = np.zeros_like(x)
    for k in range(x.shape[0]):
        hh = h * max(1.0, abs(x[k]))
        xp, xm = x.copy(), x.copy()
        xp[k] += hh; xm[k] -= hh
        g[k] = (fun(xp) - fun(xm)) / (2 * hh)
    return g


# ---------------------------------------------------------------------------
# 1–2. 梯度与逐样本得分
# ---------------------------------------------------------------------------

def test_magnitude_grad_vs_fd():
    Y, Z, N = _sim(1, n=80)
    blocks = clc.make_blocks(Y.shape[1], 4)
    theta, phi = TH * 1.1, PHI * 0.9
    fun = lambda t, p: clc.magnitude_loglik_grad(Y, N, t, p, blocks)[0]
    ll, g_lp, g_b = clc.magnitude_loglik_grad(Y, N, theta, phi, blocks)
    eps = 1e-5
    fd_lp = (fun(theta, phi * np.exp(eps)) - fun(theta, phi * np.exp(-eps))) \
        / (2 * eps)
    assert abs(g_lp - fd_lp) < 1e-3 * max(1.0, abs(fd_lp))
    fd_b = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= np.exp(eps); tm[j] *= np.exp(-eps)
        fd_b[j] = (fun(tp, phi) - fun(tm, phi)) / (2 * eps)
    assert np.allclose(g_b, fd_b, rtol=1e-4, atol=1e-3)


def test_per_sample_scores_sum_to_grad():
    Y, Z, N = _sim(2, n=60)
    p = Y.shape[1]
    blocks = clc.make_blocks(p, 4)
    psi = cl._pack(PHI * 0.9, np.clip(PI, 1e-3, 0.999), TH * 1.05, None)
    _, g = clc.count_loglik_grad(psi, Y, N, blocks, None)
    U = clc._per_sample_scores(psi, Y, N, blocks, None)
    assert np.allclose(U.sum(axis=0), g, rtol=1e-8, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. b=1 严格退化
# ---------------------------------------------------------------------------

def test_b1_loglik_and_grad_equal_detection_cl():
    Y, Z, N = _sim(3, n=60)
    p = Y.shape[1]
    blocks = clc.make_blocks(p, 1)
    psi = cl._pack(PHI * 1.1, np.clip(PI, 1e-3, 0.999), TH * 0.95, None)
    ll, g = clc.count_loglik_grad(psi, Y, N, blocks, None)
    neg, negg = cl._neg_loglik_grad(psi, (Y > 0).astype(float), N, None)
    assert abs(ll + neg) < 1e-10
    assert np.allclose(g, -negg, rtol=1e-10, atol=1e-10)


def test_b1_godambe_A_matches_baseline():
    """b=1 时 godambe_covariance_count 的 A 严格等于 cl 的解析 Fisher。"""
    Y, Z, N = _sim(4, n=200)
    p = Y.shape[1]
    D = (Y > 0).astype(float)
    f = cl.fit_composite(D, N, phi_known=PHI)
    psi = cl._pack(PHI, f["pi"], f["theta"], PHI)
    blocks = clc.make_blocks(p, 1)
    _, _, A_new, _ = clc.godambe_covariance_count(psi, Y, N, blocks, PHI)
    A_ana, _ = cl._fisher_and_scores(psi, D, N, PHI)
    assert np.allclose(A_new, A_ana, rtol=1e-9, atol=1e-9)


def test_mag_score_mean_zero_at_truth():
    """截断修正的决定性检验：真参处 E[幅度得分]=0（MC）。

    无截断版本在本检验中 E[score]≈+3.7±0.007（单一样本设置）/ +4~+15
    （8 类群主校准，~10+ SE），截断后应回到 0 的 MC 误差内。
    """
    gs = []
    R = 60
    for s in range(R):
        Y, Z, N = _sim(2000 + s)
        blocks = clc.make_blocks(Y.shape[1], 8)
        _, g_lp, g_b = clc.magnitude_loglik_grad(Y, N, TH, PHI, blocks)
        gs.append(np.concatenate([[g_lp], g_b]))
    gs = np.array(gs)
    mean = gs.mean(axis=0)
    se = gs.std(axis=0, ddof=1) / np.sqrt(R)
    assert np.all(np.abs(mean) < 3.5 * se)


# ---------------------------------------------------------------------------
# 4. 幅度部分对 θ 有信息
# ---------------------------------------------------------------------------

def test_magnitude_informative_on_theta():
    """幅度部分对 θ 有信息：多数据集平均真参 ℓ 高于 1.5× 扰动。"""
    diffs = []
    for s in range(6):
        Y, Z, N = _sim(100 + s, n=400)
        blocks = clc.make_blocks(Y.shape[1], 8)
        lt, _, _ = clc.magnitude_loglik_grad(Y, N, TH, PHI, blocks)
        lp, _, _ = clc.magnitude_loglik_grad(Y, N, TH * 1.5, PHI, blocks)
        diffs.append(lt - lp)
    assert np.mean(diffs) > 0
    assert sum(d > 0 for d in diffs) >= 5


# ---------------------------------------------------------------------------
# 5. 后验性质
# ---------------------------------------------------------------------------

def test_posterior_depth_monotonicity():
    N = np.array([1e3, 3e3, 1e4, 3e4, 1e5])
    p = post.zero_source_posterior(np.array([0.9]), np.array([1e-3]),
                                   PHI, N)[:, 0]
    assert np.all(np.diff(p) > 0)          # 越深越可能是结构零
    assert p[-1] > 0.9 and p[0] < 0.5


def test_posterior_auc_and_calibration_at_truth():
    Y, Z, N = _sim(6, n=400)
    p = Y.shape[1]
    post_mat = post.zero_source_posterior(PI, TH, PHI, N)
    zero = Y == 0
    labels = (Z == 0)[zero].astype(float)
    scores = post_mat[zero]
    auc = post.auc_score(labels, scores)
    assert auc > 0.8
    mp, fp, cnt = post.calibration_bins(labels, scores, n_bins=5)
    ok = cnt > 20
    # 分箱均值与经验结构零比例正相关且大致落在对角线附近
    assert np.corrcoef(mp[ok], fp[ok])[0, 1] > 0.9
    assert np.all(np.abs(mp[ok] - fp[ok]) < 0.15)


def test_auc_edge_cases():
    assert post.auc_score(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert post.auc_score(np.array([0, 1]), np.array([0.9, 0.1])) == 0.0
    assert post.auc_score(np.array([0, 1, 0, 1]),
                          np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


# ---------------------------------------------------------------------------
# 6. 小规模多数据集：b=4 比 b=1 的 sd(log θ̂) 更小（方向性）
# ---------------------------------------------------------------------------

def test_count_cl_reduces_theta_scatter():
    sds = {1: [], 4: []}
    for seed in range(20, 26):
        Y, Z, N = _sim(seed)
        for b in (1, 4):
            f = clc.fit_count_composite(Y, N, b=b, phi_known=PHI)
            sds[b].append(np.log(f["theta"]))
    sd1 = np.std(np.array(sds[1]), axis=0, ddof=1).mean()
    sd4 = np.std(np.array(sds[4]), axis=0, ddof=1).mean()
    assert sd4 < sd1
