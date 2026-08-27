"""composite_likelihood_cov.py — 存在层协变量的逐类群块复合似然（里程碑 1）。

模型（sec3_framework.tex 存在层，无因子、ρ≡1）：
  logit π_ij = γ_j' W_i，检出概率
  q_ij = P(D_ij = 1 | W_i, N_i) = σ(γ_j' W_i) · [1 − g(N_i; θ̄_j, φ)].

估计器：工作独立性复合似然的逐类群块——每块是一个带协变量的
"占用–检出"模型（占用率随 W_i 变化、检出率随 N_i 变化的 Bernoulli 回归）。
参数：逐类群 (γ_j ∈ R^K, β_j = log θ̄_j) + 共享 log φ（可固定为 profile）。
跨类群检出相关的校准损失由 Godambe 三明治 A⁻¹BA⁻¹ 修正（B 用逐样本得分
外积，捕获样本内跨类群相关）；区间与 Wald 检验在 γ 的自然尺度构造。

解析导数（a_ij = 1−g_ij，π_ij = σ(η_ij)，η_ij = W_i'γ_j）：
  ∂q/∂γ_jk = π(1−π)·a·W_ik
  ∂q/∂β_j  = −π·g·θ_j·∂log g/∂θ
  ∂q/∂logφ = −π·g·φ·∂log g/∂φ

参数布局 ψ：φ 未知时 ψ = (logφ, γ_1, β_1, γ_2, β_2, ..., γ_p, β_p)
（逐类群 (K+1) 维块连续存放）；φ 已知时去掉首坐标。
不修改 composite_likelihood.py 的任何公开行为；本文件为纯新增。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from model import log_g, dlogg_dtheta, dlogg_dphi

_EPS = 1e-12

__all__ = [
    "composite_loglik_cov", "fit_composite_cov", "godambe_covariance_cov",
    "wald_table", "detection_indicators",
]


# ---------------------------------------------------------------------------
# 参数打包 / 解包
# ---------------------------------------------------------------------------

def _unpack(psi, p, K, phi_known):
    """返回 (phi, Gamma (p,K), theta (p,))。"""
    off = 0 if phi_known is not None else 1
    phi = phi_known if phi_known is not None else np.exp(psi[0])
    G = psi[off:off + p * K].reshape(p, K)
    theta = np.exp(psi[off + p * K:off + p * (K + 1)])
    return phi, G, theta


def _pack(phi, Gamma, theta, phi_known):
    p, K = Gamma.shape
    parts = [Gamma.ravel(), np.log(theta)]
    if phi_known is None:
        parts = [[np.log(phi)]] + parts
    return np.concatenate([np.atleast_1d(x) for x in parts])


def _layout(p, K, phi_known):
    """各参数的指标布局。"""
    off = 0 if phi_known is not None else 1
    idx_gamma = np.arange(off, off + p * K).reshape(p, K)
    idx_beta = off + p * K + np.arange(p)
    return off, idx_gamma, idx_beta


# ---------------------------------------------------------------------------
# q 及其导数
# ---------------------------------------------------------------------------

def _q_and_grad(W, N, Gamma, theta, phi):
    """返回 q 与 ∂q/∂γ（逐协变量）、∂q/∂β、∂q/∂logφ。

    W : (n,K)；N : (n,)；Gamma : (p,K)；theta : (p,)；phi 标量。
    dq_dW : (n,p,K)，dq_dW[:,j,k] = ∂q_ij/∂γ_jk；其余 (n,p)。
    """
    eta = W @ Gamma.T                       # (n,p)
    pi = expit(eta)
    lg = log_g(N[:, None].astype(float), theta[None, :], phi)
    g = np.exp(lg)
    a = 1.0 - g
    q = pi * a
    base = (pi * (1.0 - pi) * a)            # (n,p)
    dq_dW = base[:, :, None] * W[:, None, :]        # (n,p,K)
    dq_dbeta = (-pi * g * dlogg_dtheta(N[:, None].astype(float),
                                       theta[None, :], phi)
                * theta[None, :])
    dq_dgphi = (-pi * g * dlogg_dphi(N[:, None].astype(float),
                                     theta[None, :], phi) * phi)
    return q, dq_dW, dq_dbeta, dq_dgphi


# ---------------------------------------------------------------------------
# 复合对数似然、梯度
# ---------------------------------------------------------------------------

def composite_loglik_cov(psi, D, W, N, phi_known=None):
    """复合对数似然 ℓ(ψ)（工作独立性；供检验与调试）。"""
    n, p = D.shape
    K = W.shape[1]
    phi, G, theta = _unpack(psi, p, K, phi_known)
    q, *_ = _q_and_grad(W, N, G, theta, phi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    return float((D * np.log(qc) + (1.0 - D) * np.log1p(-qc)).sum())


def _neg_loglik_grad(psi, D, W, N, phi_known=None):
    n, p = D.shape
    K = W.shape[1]
    phi, G, theta = _unpack(psi, p, K, phi_known)
    q, dqW, dqb, dqgp = _q_and_grad(W, N, G, theta, phi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    r = (D - qc) / (qc * (1.0 - qc))                # (n,p)
    ll = (D * np.log(qc) + (1.0 - D) * np.log1p(-qc)).sum()
    s_G = (r[:, :, None] * dqW).sum(axis=0)         # (p,K)
    s_b = (r * dqb).sum(axis=0)                     # (p,)
    s_gp = float((r * dqgp).sum())
    grad = np.concatenate([s_G.ravel(), s_b])
    if phi_known is None:
        grad = np.concatenate([[s_gp], grad])
    return -ll, -grad


def _fisher_and_scores(psi, D, W, N, phi_known=None):
    """A（工作模型期望 Fisher）与 U（逐样本得分 (n,k)）。"""
    n, p = D.shape
    K = W.shape[1]
    k = psi.shape[0]
    phi, G, theta = _unpack(psi, p, K, phi_known)
    q, dqW, dqb, dqgp = _q_and_grad(W, N, G, theta, phi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    w = 1.0 / (qc * (1.0 - qc))
    r = (D - qc) * w
    off, idx_G, idx_b = _layout(p, K, phi_known)

    A = np.zeros((k, k))
    U = np.zeros((n, k))
    for j in range(p):
        ia = idx_G[j]                               # (K,)
        ib = idx_b[j]
        vG = dqW[:, j, :]                           # (n,K)
        vb = dqb[:, j]
        wj = w[:, j]
        # γ_j 块
        WG = wj[:, None] * vG
        A[np.ix_(ia, ia)] = vG.T @ WG
        A[ia, ib] = A[ib, ia] = WG.T @ vb
        A[ib, ib] = float((wj * vb * vb).sum())
        U[:, ia] = r[:, j][:, None] * vG
        U[:, ib] = r[:, j] * vb
        if phi_known is None:
            vg = dqgp[:, j]
            A[0, ia] = A[ia, 0] = (wj * vg) @ vG
            A[0, ib] = A[ib, 0] = float((wj * vg * vb).sum())
    if phi_known is None:
        A[0, 0] = float((w * dqgp * dqgp).sum())
        U[:, 0] = (r * dqgp).sum(axis=1)
    return A, U


def godambe_covariance_cov(psi, D, W, N, phi_known=None, ridge=1e-10):
    """Godambe 三明治协方差 A⁻¹BA⁻¹ 与朴素 Fisher 协方差 A⁻¹。"""
    A, U = _fisher_and_scores(psi, D, W, N, phi_known)
    B = U.T @ U
    k = A.shape[0]
    A_reg = A + ridge * np.trace(A) / k * np.eye(k)
    A_inv = np.linalg.inv(A_reg)
    return A_inv @ B @ A_inv, A_inv, A, B


# ---------------------------------------------------------------------------
# 估计入口
# ---------------------------------------------------------------------------

def _default_start_cov(D, W, N, phi0=1000.0):
    """起点：γ_j 截距由平均检出率/平均检出幅度给出、斜率 0；θ₀ 取粗值。"""
    from model import g_closed
    n, p = D.shape
    K = W.shape[1]
    theta0 = np.full(p, 1e-3)
    amp = float((1.0 - g_closed(N, theta0[0], phi0)).mean())
    G0 = np.zeros((p, K))
    G0[:, 0] = logit(np.clip(D.mean(axis=0) / max(amp, 1e-6), 1e-3, 0.999))
    return G0, theta0


def _fit_single_taxon(d, W, N, phi, maxiter=300):
    """φ 固定时的单类群 (K+1) 维问题；多起点取最优。

    起点策略：斜率置 0 的矩估计起点（θ₀ 三档）+ 无协变量模型
    （composite_likelihood 的 2 维逐类群拟合）的暖起点——高 π 类群检出
    近饱和时似然脊平缓，单纯矩起点偶有漂移（validation 中量化过）。
    """
    K = W.shape[1]
    # 截距盒与基线 cl 的 π∈[1e-4,0.9999] 盒一致（logit 尺度 ±9.21）；
    # 斜率放宽到 ±15。
    b_int = (logit(1e-4), logit(0.9999))
    bounds = [b_int] + [(-15.0, 15.0)] * (K - 1) + [(np.log(1e-7), np.log(0.9))]
    best = None
    from model import g_closed
    import composite_likelihood as _cl
    mean_det = float(d.mean())
    starts = []
    for theta0 in (5e-4, 2e-3, 1e-2):
        amp = float((1.0 - g_closed(N, theta0, phi)).mean())
        s = np.zeros(K + 1)
        s[0] = logit(float(np.clip(mean_det / max(amp, 1e-6), 1e-3, 0.999)))
        s[-1] = np.log(theta0)
        starts.append(s)
    # 暖起点：无协变量 2 维拟合（截距=π̂，斜率 0）
    try:
        r0 = _cl._fit_single_taxon_phi_known(d, N, phi)
        pi0, th0 = expit(r0.x[0]), np.exp(r0.x[1])
        s = np.zeros(K + 1)
        s[0] = r0.x[0]
        s[-1] = r0.x[1]
        starts.append(s)
    except Exception:
        pass
    for s in starts:
        res = minimize(_neg_loglik_grad, s, args=(d[:, None], W, N, phi),
                       method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-13,
                                "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    if not best.success and np.max(np.abs(best.jac)) < 1e-3:
        best.success = True
        best.message = "converged (projected gradient < 1e-3)"
    return best


def fit_composite_cov(D, W, N, phi_known=None, multi_start=True, maxiter=1000):
    """存在层协变量复合似然估计。

    参数
    ----
    D         : (n, p) 0/1 检出指示
    W         : (n, K) 协变量（首列通常为 1）
    N         : (n,) 测序深度
    phi_known : 给定则固定 φ（逐类群分解为 (K+1) 维子问题）；否则联合估计。

    返回 dict：phi, Gamma (p,K), theta (p,)；se_Gamma、se_beta（Godambe）、
    se_*_naive；loglik、success、on_boundary、Wald 所需全部量。
    """
    D = np.asarray(D, dtype=float)
    W = np.asarray(W, dtype=float)
    N = np.asarray(N, dtype=float)
    n, p = D.shape
    K = W.shape[1]

    bounds_beta = (np.log(1e-7), np.log(0.9))
    b_int = (logit(1e-4), logit(0.9999))
    bounds_gamma = []
    for j in range(p):
        bounds_gamma += [b_int] + [(-15.0, 15.0)] * (K - 1)
    if phi_known is not None:
        # φ 固定：目标按类群分解，逐类群 (K+1) 维优化
        xs, funs, nits, oks = [], [], [], []
        for j in range(p):
            rj = _fit_single_taxon(D[:, j], W, N, phi_known)
            xs.append(rj.x); funs.append(rj.fun)
            nits.append(rj.nit); oks.append(bool(rj.success))
        X = np.array(xs)                            # (p, K+1)
        psi = np.concatenate([X[:, :K].ravel(), X[:, K]])

        class _Res: pass
        best = _Res()
        best.x = psi
        best.fun = float(np.sum(funs))
        best.success = all(oks)
        best.nit = int(np.sum(nits))
        best.message = "per-taxon L-BFGS-B (phi known)"
        bounds = bounds_gamma + [bounds_beta] * p
    else:
        bounds = ([(np.log(0.05), np.log(1e5))]
                  + bounds_gamma
                  + [bounds_beta] * p)
        starts = []
        for phi0 in ([1000.0, 3000.0] if multi_start else [1000.0]):
            G0, theta0 = _default_start_cov(D, W, N, phi0)
            for th0 in (np.full(p, 1e-3), theta0):
                starts.append(_pack(phi0, G0, th0, None))
        # 暖起点：φ=1000 的 profile 逐类群拟合（斜率已在其中收敛）+
        # 无协变量联合拟合（斜率 0）
        if multi_start:
            try:
                fk = fit_composite_cov(D, W, N, phi_known=1000.0,
                                       multi_start=False)
                starts.append(_pack(1000.0, fk["Gamma"], fk["theta"], None))
            except Exception:
                pass
            try:
                import composite_likelihood as _cl
                fb = _cl.fit_composite(D, N, multi_start=False)
                G1 = np.zeros((p, K))
                G1[:, 0] = logit(np.clip(fb["pi"], 1e-4, 0.9999))
                starts.append(_pack(fb["phi"], G1, fb["theta"], None))
            except Exception:
                pass
        best = None
        for s in starts:
            res = minimize(_neg_loglik_grad, s, args=(D, W, N, None),
                           method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-12,
                                    "gtol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        if not best.success and np.max(np.abs(best.jac)) < 1e-3:
            best.success = True
            best.message = "converged (projected gradient < 1e-3)"

    psi = best.x
    phi, G, theta = _unpack(psi, p, K, phi_known)
    V_god, V_naive, A, B = godambe_covariance_cov(psi, D, W, N, phi_known)
    sd_g = np.sqrt(np.maximum(np.diag(V_god), 0.0))
    sd_n = np.sqrt(np.maximum(np.diag(V_naive), 0.0))
    off, idx_G, idx_b = _layout(p, K, phi_known)

    out = {
        "phi": phi, "Gamma": G, "theta": theta,
        "loglik": -best.fun, "success": bool(best.success),
        "n_iter": int(best.nit), "message": str(best.message),
        "A": A, "B": B, "cond_A": float(np.linalg.cond(A)),
        "se_Gamma": sd_g[idx_G], "se_beta": sd_g[idx_b],
        "se_Gamma_naive": sd_n[idx_G], "se_beta_naive": sd_n[idx_b],
        "se_theta": sd_g[idx_b] * theta,
    }
    if phi_known is None:
        out["se_gamma_phi"] = sd_g[0]
        out["se_phi"] = sd_g[0] * phi
    lo = np.array([bnd[0] for bnd in bounds])
    hi = np.array([bnd[1] for bnd in bounds])
    out["on_boundary"] = (np.isclose(psi, lo, atol=1e-6)
                          | np.isclose(psi, hi, atol=1e-6))
    return out


def wald_table(fit):
    """存在性关联 Wald 检验：对 Γ 的每个元素 z = γ̂/SE(Godambe)。

    返回 dict：z (p,K)、pval (p,K)（双侧正态）。截距列是否检验由调用方
    决定（通常只检验斜率列）。
    """
    from scipy.stats import norm
    z = fit["Gamma"] / np.maximum(fit["se_Gamma"], 1e-300)
    pval = 2.0 * norm.sf(np.abs(z))
    return {"z": z, "pval": pval}


def detection_indicators(Y):
    """Y (n,p) 计数 → D (n,p) 0/1 检出指示。"""
    return (np.asarray(Y) > 0).astype(float)
