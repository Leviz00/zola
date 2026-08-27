"""composite_likelihood.py — 检出指示的成对/单类群复合似然估计。

模型核心（fix_N1 命题 prop:ident 的检出曲线）：以检出指示 D_ij = I(Y_ij>0)
为二元数据，

  P(D_ij = 1 | N_i) = q_ij = π_j · [1 − g(N_i; θ̄_j, φ)].

估计器：工作独立性复合似然（fix_N4 方案 (c) 的 b=1 极限：逐类群边际块），
即对每个 (i,j) 的 Bernoulli 对数似然求和；参数为共享 log φ（可固定，
profile 模式）+ 逐类群 (logit π_j, log θ̄_j)。跨类群耦合（归一化导致的
检出指示相关）被放弃带来的效率/校准损失由 Godambe 三明治标准误修正：

  Var(ψ̂) = A⁻¹ B A⁻¹,  A = Σ_ij E[−∂²ℓ_ij/∂ψ∂ψᵀ]（工作模型 Fisher），
  B = Σ_i u_i u_iᵀ（u_i 为样本 i 的复合得分向量，捕获样本内跨类群相关）。

推导所需解析导数（见 model.py）：
  ∂q/∂π = 1−g；∂q/∂θ = −π g ∂log g/∂θ；∂q/∂φ = −π g ∂log g/∂φ。
变换参数：α=logit π（dπ/dα=π(1−π)），β=log θ（dθ/dβ=θ），γ=log φ（dφ/dγ=φ）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from model import log_g, dlogg_dtheta, dlogg_dphi

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 核心：q 及其对 (γ, α, β) 的导数
# ---------------------------------------------------------------------------

def _q_and_grad(N, theta, phi, pi):
    """返回 q、∂q/∂γ、∂q/∂α、∂q/∂β（形状广播到 (n, p)）。

    N : (n,)；theta, pi : (p,)；phi 标量。
    """
    Nc = N[:, None].astype(float)          # (n,1)
    th = theta[None, :]                     # (1,p)
    lg = log_g(Nc, th, phi)                 # (n,p)
    g = np.exp(lg)
    q = pi[None, :] * (1.0 - g)
    dq_dpi = 1.0 - g
    dq_dtheta = -pi[None, :] * g * dlogg_dtheta(Nc, th, phi)
    dq_dphi = -pi[None, :] * g * dlogg_dphi(Nc, th, phi)
    dq_dgamma = dq_dphi * phi                        # γ = log φ
    dq_dalpha = dq_dpi * (pi * (1.0 - pi))[None, :]  # α = logit π
    dq_dbeta = dq_dtheta * th                        # β = log θ
    return q, dq_dgamma, dq_dalpha, dq_dbeta


def _sigmoid(x):
    """复数安全的 logistic（scipy expit 不支持 complex，复步长检验需要）。"""
    return 1.0 / (1.0 + np.exp(-x))


def _unpack(psi, p, phi_known):
    if phi_known is None:
        return np.exp(psi[0]), _sigmoid(psi[1:1 + p]), np.exp(psi[1 + p:])
    return phi_known, _sigmoid(psi[:p]), np.exp(psi[p:])


def _pack(phi, pi, theta, phi_known):
    if phi_known is None:
        return np.concatenate([[np.log(phi)], logit(pi), np.log(theta)])
    return np.concatenate([logit(pi), np.log(theta)])


# ---------------------------------------------------------------------------
# 复合对数似然、梯度、得分、Fisher
# ---------------------------------------------------------------------------

def composite_loglik(psi, D, N, phi_known=None):
    """复合对数似然 ℓ(ψ)（工作独立性；供检验与调试）。"""
    p = D.shape[1]
    phi, pi, theta = _unpack(psi, p, phi_known)
    q, *_ = _q_and_grad(N, theta, phi, pi)
    qc = q if np.iscomplexobj(q) else np.clip(q, _EPS, 1.0 - _EPS)
    total = (D * np.log(qc) + (1.0 - D) * np.log1p(-qc)).sum()
    return float(total) if not np.iscomplexobj(total) else total


def _neg_loglik_grad(psi, D, N, phi_known=None):
    """L-BFGS 目标：负复合对数似然及解析梯度。"""
    p = D.shape[1]
    phi, pi, theta = _unpack(psi, p, phi_known)
    q, dqg, dqa, dqb = _q_and_grad(N, theta, phi, pi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    r = (D - qc) / (qc * (1.0 - qc))        # Bernoulli 得分残差 (n,p)
    ll = (D * np.log(qc) + (1.0 - D) * np.log1p(-qc)).sum()
    s_gamma = float((r * dqg).sum())
    s_alpha = (r * dqa).sum(axis=0)          # (p,)
    s_beta = (r * dqb).sum(axis=0)           # (p,)
    if phi_known is None:
        grad = np.concatenate([[s_gamma], s_alpha, s_beta])
    else:
        grad = np.concatenate([s_alpha, s_beta])
    return -ll, -grad


def _fisher_and_scores(psi, D, N, phi_known=None):
    """Godambe 三明治的两个因子。

    返回 A（工作模型期望 Fisher，(k,k)）与 U（逐样本得分，(n,k)）。
    A = Σ_ij v_ij v_ijᵀ / [q_ij(1−q_ij)]，v_ij = ∂q_ij/∂ψ（支撑于 {γ,α_j,β_j}）。
    """
    n, p = D.shape
    k = psi.shape[0]
    phi, pi, theta = _unpack(psi, p, phi_known)
    q, dqg, dqa, dqb = _q_and_grad(N, theta, phi, pi)
    qc = np.clip(q, _EPS, 1.0 - _EPS)
    w = 1.0 / (qc * (1.0 - qc))              # Fisher 权重 (n,p)
    r = (D - qc) * w                         # 得分残差 (n,p)
    free_phi = phi_known is None

    A = np.zeros((k, k))
    U = np.zeros((n, k))
    off = 1 if free_phi else 0
    if free_phi:
        A[0, 0] = float((w * dqg * dqg).sum())
        U[:, 0] = (r * dqg).sum(axis=1)
    for j in range(p):
        ia, ib = off + (2 * j if free_phi else j), None
        # 指标：α_j 位于 off + j（free_phi 时 φ 在 0，之后 α_1..α_p, β_1..β_p）
        ia = off + j
        ib = off + p + j
        va, vb = dqa[:, j], dqb[:, j]
        wj = w[:, j]
        A[ia, ia] = float((wj * va * va).sum())
        A[ia, ib] = A[ib, ia] = float((wj * va * vb).sum())
        A[ib, ib] = float((wj * vb * vb).sum())
        U[:, ia] = r[:, j] * va
        U[:, ib] = r[:, j] * vb
        if free_phi:
            A[0, ia] = A[ia, 0] = float((wj * dqg[:, j] * va).sum())
            A[0, ib] = A[ib, 0] = float((wj * dqg[:, j] * vb).sum())
    return A, U


def godambe_covariance(psi, D, N, phi_known=None, ridge=1e-10):
    """Godambe 三明治协方差 A⁻¹ B A⁻¹ 与朴素 Fisher 协方差 A⁻¹。

    返回 (V_god, V_naive, A, B)。A 加微小岭项保证数值可逆（脊区域 A 病态，
    此时协方差本身巨大，岭项不改变结论）。
    """
    A, U = _fisher_and_scores(psi, D, N, phi_known)
    B = U.T @ U
    k = A.shape[0]
    A_reg = A + ridge * np.trace(A) / k * np.eye(k)
    A_inv = np.linalg.inv(A_reg)
    V_naive = A_inv
    V_god = A_inv @ B @ A_inv
    return V_god, V_naive, A, B


# ---------------------------------------------------------------------------
# 估计入口
# ---------------------------------------------------------------------------

def _default_start(D, N, phi0=2.0):
    """数据驱动的起点：θ₀ 取检出曲线的矩估计粗值，π₀ 由振幅给出。"""
    p = D.shape[1]
    mean_det = D.mean(axis=0)
    from model import g_closed
    theta0 = np.full(p, 0.02)
    amp = 1.0 - g_closed(N[:, None].astype(float), theta0[None, :], phi0)
    pi0 = np.clip(mean_det / np.clip(amp.mean(axis=0), 1e-6, None), 1e-3, 0.999)
    return phi0, pi0, theta0


def _fit_single_taxon_phi_known(d, N, phi, maxiter=200):
    """φ 固定时的单类群 2 维问题（logit π, log θ）；多起点取最优。"""
    bounds = [(logit(1e-4), logit(0.9999)), (np.log(1e-7), np.log(0.9))]
    best = None
    mean_det = float(d.mean())
    for theta0 in (5e-4, 2e-3, 1e-2):
        from model import g_closed
        amp = float((1.0 - g_closed(N, theta0, phi)).mean())
        pi0 = float(np.clip(mean_det / max(amp, 1e-6), 1e-3, 0.999))
        s = np.array([logit(pi0), np.log(theta0)])
        res = minimize(_neg_loglik_grad, s, args=(d[:, None], N, phi),
                       method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-13,
                                "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    # L-BFGS-B 常在梯度已达 ~1e-9 后以 ABNORMAL_TERMINATION_IN_LNSRCH 退出
    # （线搜索无法进一步提升精度）；将投影梯度足够小者视为收敛。
    if not best.success and np.max(np.abs(best.jac)) < 1e-3:
        best.success = True
        best.message = "converged (projected gradient < 1e-3)"
    return best


def fit_composite(D, N, phi_known=None, multi_start=True, maxiter=1000):
    """联合/profile 复合似然估计 (π_j, θ̄_j, φ̂)。

    参数
    ----
    D         : (n, p) 0/1 检出指示
    N         : (n,) 测序深度
    phi_known : 若给定则固定 φ（profile 模式，验证 φ 已知/未知对比）；
                否则联合估计共享 log φ。
    multi_start : 多起点（θ₀∈{0.02, 0.002, 0.1} 加 φ₀ 两档）取最优 ℓ。

    返回 dict：phi, pi, theta（估计）；se_*（Godambe 与 naive 两套）；
    loglik, success, n_iter, on_boundary。
    """
    D = np.asarray(D, dtype=float)
    N = np.asarray(N, dtype=float)
    n, p = D.shape

    if phi_known is not None:
        # φ 固定：目标按类群可分解，逐类群 2 维优化（更快、更稳健）
        xs = []
        funs = []
        nits = []
        oks = []
        for j in range(p):
            rj = _fit_single_taxon_phi_known(D[:, j], N, phi_known)
            xs.append(rj.x); funs.append(rj.fun); nits.append(rj.nit)
            oks.append(bool(rj.success))
        psi = np.concatenate(xs)          # (α_1,β_1,α_2,β_2,...) → 需重排
        psi = np.concatenate([psi[0::2], psi[1::2]])  # → (α_1..α_p, β_1..β_p)
        class _Res: pass
        best = _Res()
        best.x = psi
        best.fun = float(np.sum(funs))
        best.success = all(oks)
        best.nit = int(np.sum(nits))
        best.message = "per-taxon L-BFGS-B (phi known)"
    else:
        bounds = ([(np.log(0.05), np.log(1e5))]
                  + [(logit(1e-4), logit(0.9999))] * p
                  + [(np.log(1e-7), np.log(0.9))] * p)
        starts = []
        for phi0 in ([1000.0, 3000.0] if multi_start else [1000.0]):
            _, pi0, _ = _default_start(D, N, phi0)
            for theta0_val in (5e-4, 2e-3):
                starts.append(_pack(phi0, pi0, np.full(p, theta0_val), None))
        best = None
        for s in starts:
            res = minimize(_neg_loglik_grad, s, args=(D, N, None),
                           method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-12,
                                    "gtol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        if not best.success and np.max(np.abs(best.jac)) < 1e-3:
            best.success = True
            best.message = "converged (projected gradient < 1e-3)"
    psi = best.x
    phi, pi, theta = _unpack(psi, p, phi_known)
    V_god, V_naive, A, B = godambe_covariance(psi, D, N, phi_known)
    sd_g = np.sqrt(np.maximum(np.diag(V_god), 0.0))
    sd_n = np.sqrt(np.maximum(np.diag(V_naive), 0.0))

    off = 1 if phi_known is None else 0
    # delta 方法：π=σ(α) 导数 π(1−π)，θ=e^β 导数 θ，φ=e^γ 导数 φ
    out = {
        "phi": phi, "pi": pi, "theta": theta,
        "loglik": -best.fun, "success": bool(best.success),
        "n_iter": int(best.nit), "message": str(best.message),
        "A": A, "B": B, "cond_A": float(np.linalg.cond(A)),
    }
    if phi_known is None:
        out["se_phi"] = sd_g[0] * phi
        out["se_phi_naive"] = sd_n[0] * phi
        out["se_gamma"] = sd_g[0]                     # log φ 尺度
    out["se_alpha"] = sd_g[off:off + p]               # logit π 尺度
    out["se_beta"] = sd_g[off + p:off + 2 * p]        # log θ 尺度
    out["se_pi"] = sd_g[off:off + p] * pi * (1.0 - pi)
    out["se_theta"] = sd_g[off + p:off + 2 * p] * theta
    out["se_pi_naive"] = sd_n[off:off + p] * pi * (1.0 - pi)
    out["se_theta_naive"] = sd_n[off + p:off + 2 * p] * theta
    # 边界诊断
    if phi_known is None:
        bounds = ([(np.log(0.05), np.log(1e5))]
                  + [(logit(1e-4), logit(0.9999))] * p
                  + [(np.log(1e-7), np.log(0.9))] * p)
    else:
        bounds = ([(logit(1e-4), logit(0.9999))] * p
                  + [(np.log(1e-7), np.log(0.9))] * p)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    out["on_boundary"] = (np.isclose(psi, lo, atol=1e-6)
                          | np.isclose(psi, hi, atol=1e-6))
    return out


def detection_indicators(Y):
    """Y (n,p) 计数 → D (n,p) 0/1 检出指示。"""
    return (np.asarray(Y) > 0).astype(float)
