"""composite_likelihood_count.py — 计数幅度 DM 块复合似然（里程碑 2，评审 C1(b)）。

动机：b=1 检出指示复合似然（composite_likelihood.py）只用 D_ij=I(Y_ij>0)，
θ 层信息仅经由检出曲线 π[1−g(N;θ,φ)] 进入；计数幅度完全未用。本模块把
计数幅度以【块内归一化 DM 条件似然】加入（fix_N4 方案 (c) 的 b≥2 形态）：

  ℓ = ℓ_det(ψ) + ℓ_mag(ψ)

  ℓ_det：逐类群 Bernoulli 检出指示部分（直接复用 composite_likelihood 的
         解析梯度；b=1 时 ℓ_mag≡0，严格退化为既有估计器）；
  ℓ_mag：类群划分为大小 b 的块。对样本 i、块 B，记检出集
         S = {j∈B : Y_ij>0}、块内检出总深度 N_S = Σ_{j∈S} Y_ij。
         当 |S|≥2 时，Y_S | (S, N_S) 服从【零截断 DM】：
           P(Y_S=y | S, N_S) = DM(N_S; α_S)(y) / T_S(N_S),
           α_j = φ·θ̄_j,  A_S = Σ_{j∈S} α_j,
           T_S(N_S) = P_DM(S 中所有分量 ≥1)
                    = Σ_{U⊆S} (−1)^{|U|} g(N_S; A_U/A_S, A_S),
         其中 A_U = Σ_{j∈U} α_j，g 为 model.log_g 的闭式（折叠性质：
         DM 子集计数为零的概率 = (A_S−A_U)_{(N)}/(A_S)_{(N)} = g）。

推导要点（为什么必须是截断 DM）：条件于 z（块内在场构型）与 N_S，
Y_supp(z) | N_S ~ DM(α_supp z)（Dirichlet 中性，精确）；对 z⊇S 混合时
DM_z(y)/DM_S(y) 之比与 y 无关，故 P(y|S,N_S) ∝ DM_S(y)——但支撑被检出
事件限制在 {y_j≥1 ∀j∈S} 上，归一化常数即 T_S。漏掉 T_S 的版本（无截
断）在真参处 E[score]≠0（MC 实测 +3.7±0.007，单一样本设置），θ̂ 系
统性上偏 ~10–15% 且不随 n 消失；本实现经 E[score]=0 的 MC 检验与
R=200 自校准验证（results/count_calibration_summary.csv）。

复杂度诚实声明：T_S 的子集枚举代价 O(2^b)/样本-块（向量化 (n,2^b) 数
组），单次 ℓ_mag 评估 O(n·p·2^b/b)，故默认推荐 b≤8（b=8 时 256 项，
见 results/block_complexity.csv 实测与 fix_N4 的 O(np)/O(np·b) 声明对
照）。b=1 时恒有 ℓ_mag≡0（|S|<2 无条件组成信息），严格退化为检出指
示 CL。

解析梯度：记 t_j = ψ(α_j+Y_ij)−ψ(α_j)，T̃ = ψ(A_S)−ψ(A_S+N_S)，
  ∂ℓ_DM/∂β_l = α_l·(t_l+T̃)，∂ℓ_DM/∂logφ = Σ_j α_j(t_j+T̃)（j∈S）；
截断项：∂log g(N;θ,φ)/∂α_j = g_θ·(1[j∈U]−θ)/A + g_φ（θ=A_U/A, φ=A，
g_θ=dlogg_dtheta, g_φ=dlogg_dphi，见 model.py），∂log T_S 由子集加
权和给出。Godambe：B = 逐样本得分外积（检出部分复用 cl，幅度部分逐
样本解析）；A = 总复合似然在 ψ̂ 处对解析梯度的中心差分 Hessian
（h=1e-5，b=1 时与 cl 的解析 Fisher 对照见测试）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma, logit

import model
import composite_likelihood as cl

_EPS = 1e-12
_BMAX = 12            # 子集枚举内存上限：2^b × n 数组

__all__ = ["magnitude_loglik_scores", "count_loglik_grad",
           "composite_loglik_count", "fit_count_composite",
           "godambe_covariance_count", "make_blocks"]


# ---------------------------------------------------------------------------
# 块划分与子集枚举
# ---------------------------------------------------------------------------

def make_blocks(p, b):
    """把 p 个类群顺序划分为大小 b 的块（末块可短）。b=1 → 逐类群。"""
    b = max(1, int(b))
    return [np.arange(s, min(s + b, p)) for s in range(0, p, b)]


def _subset_table(b):
    """块内全部子集：masks (2^b, b) bool、sgn_U=(−1)^|U|、索引。缓存。"""
    M = 1 << b
    masks = ((np.arange(M)[:, None] >> np.arange(b)[None, :]) & 1).astype(bool)
    sgn = np.where(np.mod(masks.sum(axis=1), 2) == 0, 1.0, -1.0)
    return masks, sgn


def _logT_mpmath(A_val, Ns_val, A_U, sgn, valid_row, dps=50):
    """深度相消区（T≲1e-15）的 mpmath 高精度 T_S 计算。

    T = Σ_U (−1)^|U| g(Ns; A_U/A, A)，log g 用 mp.loggamma 闭式。
    仅对被标记的样本行调用（优化器在病态角落的瞬态评估）；返回 float。
    """
    import mpmath as mp
    mp.mp.dps = dps
    A_m = mp.mpf(A_val)
    N_m = mp.mpf(Ns_val)
    T = mp.mpf(0)
    for u in np.where(valid_row)[0]:
        AU = A_U[u]
        a = A_m - mp.mpf(AU)
        # a≤0：U⊇S（g=0）或浮点噪声使 A_U 略超 A（g≈0）；均跳过。
        # （mp.loggamma 对负参返回复数，会污染求和。）
        if a <= 0:
            continue
        lg = (mp.loggamma(a + N_m) - mp.loggamma(a)
              + mp.loggamma(A_m) - mp.loggamma(A_m + N_m))
        T += mp.mpf(float(sgn[u])) * mp.e ** lg
    if T <= 0:
        return -745.0
    return float(mp.log(T))


# ---------------------------------------------------------------------------
# 计数幅度部分：对数似然 + 逐样本得分（π 不进入；条件于检出）
# 返回 (ll, Smag)：Smag (n, 1+p)，列序 (logφ, β_1..β_p)。
# ---------------------------------------------------------------------------

def magnitude_loglik_scores(Y, N, theta, phi, blocks):
    """零截断 DM 块条件对数似然与逐样本得分矩阵。"""
    theta = np.asarray(theta, dtype=float)
    p = theta.shape[0]
    n = Y.shape[0]
    alpha = phi * theta                              # (p,)
    Smag = np.zeros((n, 1 + p))
    ll = 0.0
    Yf = np.asarray(Y, dtype=float)
    for blk in blocks:
        b = blk.size
        if b < 2:
            continue
        if b > _BMAX:
            raise ValueError("块大小 b>%d 的子集枚举超出内存预算" % _BMAX)
        Yb = Yf[:, blk]                              # (n, b)
        ab = alpha[blk]                              # (b,)
        det0 = Yb > 0
        ns = det0.sum(axis=1)
        use = ns >= 2
        if not np.any(use):
            continue
        iu = np.where(use)[0]
        nu = iu.shape[0]
        det = det0[iu]                               # (nu, b)
        Yd = np.where(det, Yb[iu], 0.0)
        A = det @ ab                                 # (nu,)
        Ns = Yd.sum(axis=1)                          # (nu,)

        # ---- DM 主项 --------------------------------------------------
        Tsh = digamma(A) - digamma(A + Ns)           # (nu,)
        t_gam = gammaln(ab[None, :] + Yd) - gammaln(ab)[None, :]
        ll_b = gammaln(A) - gammaln(A + Ns) + t_gam.sum(axis=1)
        dt = digamma(ab[None, :] + Yd) - digamma(ab)[None, :]
        hpos = (dt + Tsh[:, None]) * det             # ∂ℓ_DM/∂α
        sc_pos = ab[None, :] * hpos                  # (nu,b) β 坐标得分
        # ----------------------------------------------------------------

        # ---- 截断项 T_S（子集枚举）------------------------------------
        masks, sgn = _subset_table(b)                # (M,b), (M,)
        M = masks.shape[0]
        A_U = masks @ ab                             # (M,)
        # 裁剪到 [0,1]：浮点求和顺序可使 thM 略超 1（→ gammaln(负) = nan）；
        # thM=1（U=S）时 g=0、log g=−inf，该项自然消失。
        thM = np.clip(A_U[None, :] / A[:, None], 0.0, 1.0)
        AM = A[:, None] * np.ones((1, M))
        lgM = model.log_g(Ns[:, None], thM, AM)      # (nu,M)
        gM = np.exp(lgM)
        valid = (~(masks[None, :, :] & ~det[:, None, :])).all(axis=2)
        wM = valid * sgn[None, :]                    # (nu,M)
        # T_S = Σ_U (−1)^|U| g_U：交替和在 T 很小时灾难性相消，改用
        # 符号 logsumexp：T = exp(L+)·(−expm1(L−−L+))，L± 为偶/奇子集
        # 的 logsumexp（expm1 在 L−−L+→0 时保持精度）。
        from scipy.special import logsumexp
        even = sgn > 0
        lge = logsumexp(np.where(valid & even[None, :], lgM, -np.inf),
                        axis=1)
        lgo = logsumexp(np.where(valid & ~even[None, :], lgM, -np.inf),
                        axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            logT = lge + np.log(-np.expm1(lgo - lge))
        # 深度相消标记：expm1 失效（nan）或 L+−L−>15（T<exp(L+)·3e-7，
        # float64 不可信）。这些行用 mpmath 高精度重算 logT——仅出现在
        # α 极端失衡的病态角落（优化器瞬态），正常区域无触发、零代价。
        flag = (~np.isfinite(logT)) | ((lge - lgo) > 15.0) | (logT < -100.0)
        if np.any(flag):
            idx = np.where(flag)[0]
            for r in idx:
                logT[r] = _logT_mpmath(A[r], Ns[r], A_U, sgn, valid[r])
        ll_b -= logT

        # ∂log T/∂α_j：log g(N;θ,φ)（θ=A_U/A, φ=A）对 α_j 的导数。
        # U⊇S 时 θ=1、g=0（digamma(0) 奇点）：该项对 T 与梯度均贡献 0，
        # 用截断 θ 计算导数再经 wgt(=0) 置零，避免 inf×0=nan。
        thC = np.clip(thM, 0.0, 1.0 - 1e-12)
        dgt = model.dlogg_dtheta(Ns[:, None], thC, AM)
        dgp = model.dlogg_dphi(Ns[:, None], thC, AM)
        # ∂logT/∂α_j = Σ_U (−1)^|U| exp(lgM_U − logT)·∂log g_U/∂α_j
        # （log 空间归一权重，无 0/0 问题；logT 已被 mpmath 修正）。
        wn = wM * np.exp(np.clip(lgM - logT[:, None], None, 50.0))
        dlogT = np.zeros((nu, b))
        for j in range(b):
            mj = masks[:, j][None, :]                # (1,M)
            dlogg_daj = dgt * (mj - thM) / A[:, None] + dgp
            dlogT[:, j] = (wn * dlogg_daj).sum(axis=1)
        # α_j 仅在 j∈S（检出）时进入 T_S 与 DM；未检出坐标导数恒 0
        dlogT = dlogT * det
        sc_neg = ab[None, :] * dlogT
        # ----------------------------------------------------------------

        ll += float(ll_b.sum())
        sc = (sc_pos - sc_neg)                       # (nu,b)
        Smag[np.ix_(iu, 1 + blk)] += sc
        Smag[iu, 0] += sc.sum(axis=1)                # logφ 得分
    return ll, Smag


def magnitude_loglik_grad(Y, N, theta, phi, blocks):
    """(ll, g_logphi, g_beta)：magnitude_loglik_scores 的聚合形式。"""
    ll, Smag = magnitude_loglik_scores(Y, N, theta, phi, blocks)
    s = Smag.sum(axis=0)
    return ll, float(s[0]), s[1:]


def count_loglik_grad(psi, Y, N, blocks, phi_known=None,
                      prior_eta0=np.log(50.0), prior_lam=0.0):
    """总复合对数似然 ℓ_det+ℓ_mag 与全参数解析梯度。

    v2：prior_lam>0 时加入 log-φ 二次惩罚 −λ/2·(η−η₀)²（η=psi[0]=logφ，
    仅 φ 自由时），梯度项 −λ(η−η₀)（η 即优化坐标，无链式外因子）。
    prior_lam=0 严格退化为 v1 行为。
    """
    p = Y.shape[1]
    D = (Y > 0).astype(float)
    neg_det, neg_g = cl._neg_loglik_grad(psi, D, N, phi_known)
    phi, pi, theta = cl._unpack(psi, p, phi_known)
    ll_m, g_lp, g_b = magnitude_loglik_grad(Y, N, theta, phi, blocks)
    ll = -neg_det + ll_m
    grad = -neg_g
    off = 0 if phi_known is not None else 1
    grad[off + p:off + 2 * p] += g_b
    if phi_known is None:
        grad[0] += g_lp
        if prior_lam > 0.0:
            ll -= 0.5 * prior_lam * (psi[0] - prior_eta0) ** 2
            grad[0] -= prior_lam * (psi[0] - prior_eta0)
    return ll, grad


def composite_loglik_count(psi, Y, N, blocks, phi_known=None):
    """总复合对数似然（供检验与调试）。"""
    ll, _ = count_loglik_grad(psi, Y, N, blocks, phi_known)
    return ll


# ---------------------------------------------------------------------------
# Godambe：A = 总 CL 对解析梯度的中心差分 Hessian；B = 逐样本得分外积
# ---------------------------------------------------------------------------

def _mag_hessian_fd(theta, phi, Y, N, blocks, rel_h=1e-5):
    """幅度部分 ℓ_mag 对 (logφ, β) 的中心差分 Hessian（∂²ℓ_mag）。

    返回 (1+p, 1+p)，行/列序 (logφ, β_1..β_p)。仅幅度部分数值化
    （p+1 维，成本低）；检出部分的 A 用 cl 的解析期望 Fisher。
    """
    k = 1 + theta.shape[0]

    def grad(x):
        _, g_lp, g_b = magnitude_loglik_grad(Y, N, np.exp(x[1:]),
                                             np.exp(x[0]), blocks)
        return np.concatenate([[g_lp], g_b])

    x0 = np.concatenate([[np.log(phi)], np.log(theta)])
    H = np.zeros((k, k))
    for c in range(k):
        h = rel_h * max(1.0, abs(x0[c]))
        xp, xm = x0.copy(), x0.copy()
        xp[c] += h; xm[c] -= h
        H[:, c] = (grad(xp) - grad(xm)) / (2 * h)
    return 0.5 * (H + H.T)


def _per_sample_scores(psi, Y, N, blocks, phi_known):
    """逐样本得分矩阵 U (n,k)：检出部分复用 cl，幅度部分逐样本解析。"""
    p = Y.shape[1]
    D = (Y > 0).astype(float)
    _, U = cl._fisher_and_scores(psi, D, N, phi_known)
    phi, pi, theta = cl._unpack(psi, p, phi_known)
    off = 0 if phi_known is not None else 1
    _, Smag = magnitude_loglik_scores(Y, N, theta, phi, blocks)
    U[:, off + p:off + 2 * p] += Smag[:, 1:]
    if phi_known is None:
        U[:, 0] += Smag[:, 0]
    return U


def godambe_covariance_count(psi, Y, N, blocks, phi_known=None, ridge=1e-10):
    """Godambe 三明治协方差。

    A = A_det（检出部分解析期望 Fisher，复用 cl._fisher_and_scores）
        + (−H_mag)（幅度部分观测 Hessian，_mag_hessian_fd 中心差分）。
    B = 逐样本得分外积（检出复用 cl，幅度逐样本解析）。
    b=1 时 ℓ_mag≡0，A 严格等于 cl 的解析 Fisher。
    返回 (V_god, V_naive, A, B)。
    """
    p = Y.shape[1]
    D = (Y > 0).astype(float)
    A, _ = cl._fisher_and_scores(psi, D, N, phi_known)
    phi, pi, theta = cl._unpack(psi, p, phi_known)
    Hm = _mag_hessian_fd(theta, phi, Y, N, blocks)
    off = 0 if phi_known is not None else 1
    if phi_known is None:
        A[0, 0] -= Hm[0, 0]
        A[0, off + p:off + 2 * p] -= Hm[0, 1:]
        A[off + p:off + 2 * p, 0] -= Hm[1:, 0]
    A[off + p:off + 2 * p, off + p:off + 2 * p] -= Hm[1:, 1:]
    k = A.shape[0]
    scale = np.trace(np.abs(A)) / k
    A_reg = A + ridge * scale * np.eye(k)
    U = _per_sample_scores(psi, Y, N, blocks, phi_known)
    B = U.T @ U
    A_inv = np.linalg.pinv(A_reg)
    return A_inv @ B @ A_inv, A_inv, A, B


# ---------------------------------------------------------------------------
# 估计入口
# ---------------------------------------------------------------------------

def fit_count_composite(Y, N, b=4, blocks=None, phi_known=None, maxiter=1000,
                        prior_eta0=np.log(50.0), prior_lam=0.44):
    """计数幅度 DM 块复合似然估计（v2：log-φ 弱信息先验，默认开启）。

    参数
    ----
    Y         : (n, p) 计数
    N         : (n,) 测序深度
    b         : 块大小（b=1 时严格退化为 cl.fit_composite 的检出指示 CL；
              b≤8 推荐，b≤12 上限——截断子集枚举 O(2^b)/样本-块）
    blocks    : 显式块划分（覆盖 b）
    phi_known : 给定则固定 φ

    返回 dict：phi, pi, theta；se_*（Godambe）；loglik（总 CL）、
    loglik_det、loglik_mag、success、on_boundary。
    """
    Y = np.asarray(Y, dtype=float)
    N = np.asarray(N, dtype=float)
    n, p = Y.shape
    if blocks is None:
        blocks = make_blocks(p, b)

    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(logit(1e-4), logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)
    if phi_known is not None:
        bounds = bounds[1:]

    def obj(psi):
        ll, g = count_loglik_grad(psi, Y, N, blocks, phi_known,
                                  prior_eta0, prior_lam)
        # 病态角落（α 极端失衡、T~1e-95）的梯度达 ~1e308，会毒化
        # L-BFGS-B 线搜索的三次插值；裁剪 ∞-范数保持方向、保住线搜索。
        # 真最优附近梯度 ≪1e3，不受影响。
        ng = float(np.abs(g).max())
        if ng > 1e3:
            g = g * (1e3 / ng)
        return -ll, -g

    # 起点：检出指示 CL 的拟合（ℓ_det 部分的最优点）+ 多起点
    starts = []
    f_det = cl.fit_composite(Y > 0, N, phi_known=phi_known,
                             multi_start=True, maxiter=maxiter)
    starts.append(f_det_psi(f_det, p, phi_known))
    for phi0 in ([1000.0, 3000.0] if phi_known is None else []):
        _, pi0, _ = cl._default_start(Y > 0, N, phi0)
        starts.append(cl._pack(phi0, pi0, np.full(p, 2e-3), None))
    best = None
    for s in starts:
        res = minimize(obj, s, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    if not best.success and np.max(np.abs(best.jac)) < 1e-3:
        best.success = True
        best.message = "converged (projected gradient < 1e-3)"

    psi = best.x
    phi, pi, theta = cl._unpack(psi, p, phi_known)
    ll_m, _, _ = magnitude_loglik_grad(Y, N, theta, phi, blocks)
    V_god, V_naive, A, B = godambe_covariance_count(psi, Y, N, blocks,
                                                    phi_known)
    sd_g = np.sqrt(np.maximum(np.diag(V_god), 0.0))
    sd_n = np.sqrt(np.maximum(np.diag(V_naive), 0.0))
    off = 1 if phi_known is None else 0

    hit_maxiter = ("ITERATIONS" in str(best.message).upper()
                   or int(best.nit) >= maxiter)
    out = {
        "phi": phi, "pi": pi, "theta": theta,
        "loglik": -best.fun, "loglik_mag": ll_m,
        "loglik_det": -best.fun - ll_m,
        "success": bool(best.success),
        "converged": bool(best.success and not hit_maxiter),
        "hit_maxiter": bool(hit_maxiter),
        "n_iter": int(best.nit),
        "prior_eta0": float(prior_eta0), "prior_lam": float(prior_lam),
        "message": str(best.message), "b": b, "n_blocks": len(blocks),
        "A": A, "B": B, "cond_A": float(np.linalg.cond(A)),
        "se_alpha": sd_g[off:off + p], "se_beta": sd_g[off + p:off + 2 * p],
        "se_pi": sd_g[off:off + p] * pi * (1 - pi),
        "se_theta": sd_g[off + p:off + 2 * p] * theta,
        "se_alpha_naive": sd_n[off:off + p],
        "se_beta_naive": sd_n[off + p:off + 2 * p],
    }
    if phi_known is None:
        out["se_gamma_phi"] = sd_g[0]
        out["se_phi"] = sd_g[0] * phi
    lo = np.array([bnd[0] for bnd in bounds])
    hi = np.array([bnd[1] for bnd in bounds])
    out["on_boundary"] = (np.isclose(psi, lo, atol=1e-6)
                          | np.isclose(psi, hi, atol=1e-6))
    out["phi_on_boundary"] = (bool(out["on_boundary"][0])
                              if phi_known is None else False)
    return out


def f_det_psi(f_det, p, phi_known):
    """把 cl.fit_composite 的返回字典还原为 ψ 起点。"""
    return cl._pack(f_det["phi"], f_det["pi"], f_det["theta"], phi_known)
