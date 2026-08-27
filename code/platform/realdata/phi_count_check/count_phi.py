"""count_phi.py — 计数幅度侧的逐类群 ZIBB 最大似然 φ 估计（任务 B 核心模块）。

模型（与检出曲线估计器逐类群边际一致，fix_N1 §3.1 ZIBB）：
  Z_ij ~ Bern(ν_j)；p_ij | Z=1 ~ Beta(φθ, φ(1−θ))；Y_ij | Z=1 ~ Binom(N_i, p_ij)
  ll = Σ_{y=0} log[(1−ν) + ν·BB(0; N,a,b)] + Σ_{y>0} [log ν + log BB(y; N,a,b)]
  BB(y; N, a=φθ, b=φ(1−θ)) = C(N,y) B(y+a, N−y+b)/B(a,b)

与检出曲线 φ̂ 的区别：同一边际参数，但识别来源是**计数幅度**（零的过剩 +
正计数方差），而非二值检出随深度的曲线形状。逐类群 φ_j（非共享）→
分布汇总（中位数/IQR）与检出侧共享 φ̂=1454 对比。

数值：log φ、logit ν、log θ 参数化，L-BFGS-B 多起点；SE(log φ) 由中心
差分 Hessian 数值二阶导给出（3 参数，代价可忽略）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, betaln, expit, logit

NU_B = (logit(1e-6), logit(0.999999))
TH_B = (np.log(1e-8), np.log(0.95))
PHI_B = (np.log(0.05), np.log(1e7))


def zibb_nll(psi, y, N):
    """负对数似然。psi = (logit ν, log θ, log φ)。"""
    nu, th, phi = expit(psi[0]), np.exp(psi[1]), np.exp(psi[2])
    a, b = phi * th, phi * (1.0 - th)
    N = N.astype(float)
    pos = y > 0
    ll = 0.0
    # 正计数
    yp, Np = y[pos], N[pos]
    ll += np.sum(np.log(nu) + gammaln(Np + 1) - gammaln(yp + 1)
                 - gammaln(Np - yp + 1)
                 + betaln(yp + a, Np - yp + b) - betaln(a, b))
    # 零
    Nz = N[~pos]
    logbb0 = betaln(a, Nz + b) - betaln(a, b)
    ll += np.sum(np.logaddexp(np.log1p(-nu), np.log(nu) + logbb0))
    return -ll


def fit_taxon_zibb(y, N, phi0_grid=(300.0, 3000.0, 3e4)):
    """逐类群 ZIBB MLE + 数值 Hessian SE(log φ)。多起点取最优。"""
    y = np.asarray(y, dtype=float)
    N = np.asarray(N, dtype=float)
    prev = float((y > 0).mean())
    pos = y > 0
    th_data = float(np.median(y[pos] / N[pos])) if pos.any() else 1e-4
    nu0 = float(np.clip(prev * 1.15, 1e-3, 0.999))
    best = None
    for phi0 in phi0_grid:
        for th0 in (th_data, th_data * 0.3, th_data * 3.0):
            s = np.array([logit(nu0), np.log(np.clip(th0, 1e-8, 0.9)),
                          np.log(phi0)])
            r = minimize(zibb_nll, s, args=(y, N), method="L-BFGS-B",
                         bounds=[NU_B, TH_B, PHI_B],
                         options={"maxiter": 500, "ftol": 1e-12})
            if best is None or r.fun < best.fun:
                best = r
    x = best.x
    # 数值 Hessian（中心差分，log 尺度步长）
    h = 1e-4
    H = np.zeros((3, 3))
    f0 = best.fun
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3); ej = np.zeros(3)
            ei[i] = h; ej[j] = h
            if i == j:
                H[i, i] = (zibb_nll(x + ei, y, N) - 2 * f0
                           + zibb_nll(x - ei, y, N)) / h ** 2
            else:
                H[i, j] = (zibb_nll(x + ei + ej, y, N)
                           - zibb_nll(x + ei - ej, y, N)
                           - zibb_nll(x - ei + ej, y, N)
                           + zibb_nll(x - ei - ej, y, N)) / (4 * h ** 2)
                H[j, i] = H[i, j]
    try:
        cov = np.linalg.inv(H + 1e-10 * np.eye(3))
        se_logphi = float(np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        se_logphi = np.nan
    on_b = (np.isclose(x[2], PHI_B[0], atol=1e-3)
            or np.isclose(x[2], PHI_B[1], atol=1e-3))
    return {"nu": float(expit(x[0])), "theta": float(np.exp(x[1])),
            "phi": float(np.exp(x[2])), "se_logphi": se_logphi,
            "loglik": float(-best.fun), "success": bool(best.success),
            "phi_on_boundary": bool(on_b)}
