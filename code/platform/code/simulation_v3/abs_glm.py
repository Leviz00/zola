"""abs_glm.py — v3.2b 绝对丰度检验尺 abs_nb_glm。

逐类群 NB2 GLM（statsmodels discrete NegativeBinomial，α 随 β 联合 MLE）:
    Y_ij ~ NB(mu_ij, alpha_j),  log mu_ij = b0 + b1 * group_i + log N_i
默认对 b1 做**似然比检验（LRT）**，跨类群 BH。Wald 在小 n + 重过离散格
（bridge 1008: n=50, φ=3）反保守（FDR 0.28）且 zinb null 下 FWER 0.20；
LRT 在相同数据上全部校准到 0.00-0.055（V323_MEMO §3 数字）。

设计要点（见 method_fix/v3/V323_MEMO.md）：
* offset 的 N_i 是**设计深度**（绝对尺的外部锚，实战中对应 spike-in 标定）。
  absolute 生成模式下病例组实测库容自然上浮；若用实测总库容做 offset，
  会把组成性重新引入（所有非 DA 类群获得 −log(uplift) 的伪组效应）。
* α_j 必须联合 MLE：固定矩估计 α 的 GLM 在本网格的重过离散 + 潜层
  Dirichlet 混合下系统偏小，null 下 |z|>1.96 比例 0.19；联合 MLE 校准到
  ≈0.044（V323_MEMO §3 有数字）。
* W（可选，(2n, p)）：结构零剔除掩码，W_ij > 0.5 的细胞进入拟合
  （oracle=presence；估计权重可阈值化）。informative 结构类群病例组全被
  剔 -> 组列恒定 -> p=1（存在层差异不属于绝对丰度 DA，与 abs_da_truth
  语义一致）。
* 防分离护栏：每组非零计数 < min_nonzero 的类群 p=1（近分离会产生
  |b1| 巨大的伪显著）。

验收（V323_MEMO §3）：全局 null 下 FDR≈0.05（R=20 快筛）；absolute 模式
DA 格 oracle 掩码注入后 FDR≈0.05。
"""
from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

__all__ = ["abs_nb_glm", "abs_nb_lrt_stats"]


def abs_nb_lrt_stats(Y, group, N=None, W=None, min_nonzero=3, maxiter=200):
    """逐类群 NB2 GLM 的 LRT 统计量（供置换经验校准，v3.6）。

    与 abs_nb_glm(test="lrt") 完全相同的拟合路径/护栏，但返回原始 LRT
    统计量而非渐近 p 值。新增函数，不改变 abs_nb_glm 的任何行为。

    Returns dict(lrt=(p,) nan=fallback, b1, alpha_hat, fallback=(p,) bool,
                 n_fallback)
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group, dtype=float)
    n, p = Y.shape
    if N is None:
        N = Y.sum(axis=1)
    N = np.asarray(N, dtype=float)
    off_all = np.log(np.maximum(N, 1.0))
    X = np.column_stack([np.ones(n), group])

    lrt = np.full(p, np.nan)
    b1 = np.full(p, np.nan)
    alpha_hat = np.full(p, np.nan)
    fallback = np.ones(p, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(p):
            if W is not None:
                keep = np.asarray(W[:, j]) > 0.5
            else:
                keep = np.ones(n, dtype=bool)
            y = Y[keep, j]
            g = group[keep]
            if (keep.sum() < 10 or y.sum() == 0 or np.unique(g).size < 2
                    or (y[g == 0] > 0).sum() < min_nonzero
                    or (y[g == 1] > 0).sum() < min_nonzero):
                continue
            try:
                mod = sm.NegativeBinomial(y, X[keep], offset=off_all[keep],
                                          loglike_method="nb2")
                res = mod.fit(disp=0, maxiter=maxiter)
                if not res.mle_retvals.get("converged", True):
                    continue
                red = sm.NegativeBinomial(y, X[keep][:, [0]],
                                          offset=off_all[keep],
                                          loglike_method="nb2")
                rres = red.fit(disp=0, maxiter=maxiter)
                lrt[j] = max(0.0, -2.0 * (rres.llf - res.llf))
                b1[j] = res.params[1]
                alpha_hat[j] = float(np.exp(res.params[-1]))
                fallback[j] = False
            except Exception:
                continue
    return dict(lrt=lrt, b1=b1, alpha_hat=alpha_hat, fallback=fallback,
                n_fallback=int(fallback.sum()))


def _bh(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, m + 1) / m
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])
        reject[order[:k + 1]] = True
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty(m)
    qvals[order] = np.clip(q, 0.0, 1.0)
    return reject, qvals


def abs_nb_glm(Y, group, N=None, W=None, alpha=0.05, min_nonzero=3,
               maxiter=200, test="lrt"):
    """Per-taxon NB2 GLM test (LRT default, Wald optional) + BH.

    Parameters
    ----------
    Y     : (2n, p) counts
    group : (2n,) 0/1
    N     : (2n,) design depths for the offset; None -> realised totals
            (NOT recommended under effect_mode='absolute': reintroduces
            compositionality; kept for legacy-mode convenience).
    W     : optional (2n, p) keep-mask (>0.5 keeps the cell)
    alpha : nominal FDR level
    min_nonzero : per-group minimum nonzero counts (separation guard)
    test  : "lrt" (default, reduced-model likelihood ratio, 1 df) or "wald"

    Returns dict(reject, pvals, qvals, b1, alpha_hat, n_fallback)
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group, dtype=float)
    n, p = Y.shape
    if N is None:
        N = Y.sum(axis=1)
        warnings.warn("abs_nb_glm: N=None uses realised totals as offset; "
                      "under effect_mode='absolute' this reintroduces "
                      "compositionality. Pass design depths.")
    N = np.asarray(N, dtype=float)
    off_all = np.log(np.maximum(N, 1.0))
    X = np.column_stack([np.ones(n), group])

    pvals = np.ones(p)
    b1 = np.full(p, np.nan)
    alpha_hat = np.full(p, np.nan)
    n_fallback = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(p):
            if W is not None:
                keep = np.asarray(W[:, j]) > 0.5
            else:
                keep = np.ones(n, dtype=bool)
            y = Y[keep, j]
            g = group[keep]
            if (keep.sum() < 10 or y.sum() == 0 or np.unique(g).size < 2
                    or (y[g == 0] > 0).sum() < min_nonzero
                    or (y[g == 1] > 0).sum() < min_nonzero):
                n_fallback += 1
                continue
            try:
                mod = sm.NegativeBinomial(y, X[keep], offset=off_all[keep],
                                          loglike_method="nb2")
                res = mod.fit(disp=0, maxiter=maxiter)
                if not res.mle_retvals.get("converged", True):
                    n_fallback += 1
                    continue
                b1[j] = res.params[1]
                alpha_hat[j] = float(np.exp(res.params[-1]))
                if test == "wald":
                    pvals[j] = 2.0 * norm.sf(abs(res.params[1] / res.bse[1]))
                else:
                    red = sm.NegativeBinomial(y, X[keep][:, [0]],
                                              offset=off_all[keep],
                                              loglike_method="nb2")
                    rres = red.fit(disp=0, maxiter=maxiter)
                    from scipy.stats import chi2
                    lr = max(0.0, -2.0 * (rres.llf - res.llf))
                    pvals[j] = float(chi2.sf(lr, 1))
            except Exception:
                n_fallback += 1
                continue
    reject, qvals = _bh(pvals, alpha)
    return dict(reject=reject, pvals=pvals, qvals=qvals, b1=b1,
                alpha_hat=alpha_hat, n_fallback=n_fallback)
