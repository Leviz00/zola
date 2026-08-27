"""posterior.py — 逐单元零来源后验（eq:posterior 的逐单元近似实现，里程碑 2）。

sec3_framework.tex 式 eq:posterior：
  Pr(Z_ij=0 | Y_i, N_i) = (1−π_ij) M_i(0) / [(1−π_ij) M_i(0) + π_ij M_i(1)]

其中 M_i(z) 为对 Z_{i,−j} 与 (θ_i, ρ_i) 边际化后的计数似然。精确求和需
遍历 2^{p−1} 构型（tex 明言不可行）；本模块实现 tex 自己给出的【逐单元
近似】：ρ≡1、其余类群固定在在场（对应主校准的 π_bulk≈1 设计，E1≤0.3%
已由 e1_check.csv 量化），此时
  M_i(1) = Pr(Y_ij=0 | Z_ij=1, N_i) = g(N_i; θ̂_j, φ̂)，  M_i(0) = 1，
  post_ij = (1−π_ij) / [(1−π_ij) + π_ij·g(N_i; θ̂_j, φ̂)]    （Y_ij=0 的单元）

Y_ij>0 的单元 Pr(Z_ij=0)=0（检出即在场）。后验显式依赖深度 N_i：同一观
测零在深样本中更可能是结构零——这是 eq:posterior 下计算讨论的核心性质，
在 validate_count.py 中以 AUC/分组校准验证。

不修改既有文件；本文件为纯新增。
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from model import log_g

__all__ = ["zero_source_posterior", "zero_source_posterior_cov",
           "auc_score", "calibration_bins"]


def zero_source_posterior(pi, theta, phi, N):
    """逐单元零来源后验 Pr(Z_ij=0 | Y_ij=0)（对所有单元返回值）。

    pi : (p,) 或 (n,p)；theta : (p,)；phi 标量；N : (n,)。
    返回 (n,p)：每个单元的条件后验；仅对 Y_ij=0 的单元有意义
    （Y_ij>0 时 Pr(Z_ij=0)=0，调用方负责掩码）。
    """
    pi = np.asarray(pi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    N = np.asarray(N, dtype=float)
    if pi.ndim == 1:
        pi = np.broadcast_to(pi[None, :], (N.shape[0], pi.shape[0]))
    g = np.exp(log_g(N[:, None], theta[None, :], phi))
    num = 1.0 - pi
    return num / (num + pi * g)


def zero_source_posterior_cov(Gamma, W, theta, phi, N):
    """协变量版：π_ij = σ(γ_j'W_i)。Gamma (p,K)，W (n,K)。"""
    pi = expit(np.asarray(W) @ np.asarray(Gamma).T)
    return zero_source_posterior(pi, theta, phi, N)


def auc_score(labels, scores):
    """Mann–Whitney AUC（含并列处理的秩统计量），labels ∈ {0,1}。"""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0])
    sr = scores[order]
    # 平均秩处理并列
    i = 0
    r = 0
    sorted_ranks = np.empty(scores.shape[0])
    while i < scores.shape[0]:
        j = i
        while j + 1 < scores.shape[0] and sr[j + 1] == sr[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        sorted_ranks[i:j + 1] = avg
        i = j + 1
    ranks[order] = sorted_ranks
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def calibration_bins(labels, scores, n_bins=10):
    """可靠性表：按预测分数等宽分箱，返回 (mean_pred, frac_pos, count)。

    labels ∈ {0,1}（1 = 真结构零），scores = 后验概率。
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        m = idx == k
        count[k] = m.sum()
        if m.any():
            mean_pred[k] = scores[m].mean()
            frac_pos[k] = labels[m].mean()
    return mean_pred, frac_pos, count
