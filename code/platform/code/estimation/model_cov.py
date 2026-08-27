"""model_cov.py — 存在层协变量扩展的模拟器（里程碑 1，评审 C1(a)）。

权威来源：outline_src/sections/sec3_framework.tex 存在层
  logit π_ij = γ_j' W_i + u_j' λ_i
本模块实现【协变量版、无因子】：u_j ≡ 0，即
  Z_ij ~ Bernoulli(π_ij),  logit π_ij = γ_j' W_i
组成层与测量层与 model.py 基线完全一致（Dirichlet(φ θ̄)、屏蔽-归一、多项），
ρ≡1。因子结构 λ_i、检出效率层 ρ 仍不实现（README 已知局限）。

不修改 model.py 的任何公开行为；本文件为纯新增。
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_three_layer_cov"]


def simulate_three_layer_cov(Gamma, W, theta_bar, phi, N, rng):
    """存在层带协变量的三层生成器（ρ≡1、无因子、单一精度 φ）。

    参数
    ----
    Gamma     : (p, K) 逐类群存在层系数 γ_j（含截距列）
    W         : (n, K) 样本协变量（首列通常为 1 截距）
    theta_bar : (p,) 基线组成 θ̄，须 Σ_j θ̄_j = 1
    phi       : 标量，合并后的单一精度
    N         : (n,) 正整数测序深度
    rng       : numpy Generator

    返回
    ----
    Y : (n, p) int 计数矩阵；Z : (n, p) bool 存在指示
    """
    Gamma = np.asarray(Gamma, dtype=float)
    W = np.asarray(W, dtype=float)
    theta_bar = np.asarray(theta_bar, dtype=float)
    N = np.asarray(N, dtype=int)
    assert np.isclose(theta_bar.sum(), 1.0), "theta_bar 须归一"
    p, K = Gamma.shape
    assert W.shape[1] == K, "W 与 Gamma 的协变量维数不一致"
    n = N.shape[0]
    assert W.shape[0] == n

    # 存在层：logit π_ij = γ_j' W_i
    eta = W @ Gamma.T                       # (n, p)
    Pi = 1.0 / (1.0 + np.exp(-eta))
    Z = rng.random((n, p)) < Pi

    # 组成层 + 屏蔽归一 + 多项测量层（与 model.simulate_three_layer 相同）
    Theta = rng.dirichlet(phi * theta_bar, size=n)
    C = Theta * Z
    S = C.sum(axis=1, keepdims=True)
    empty = (S[:, 0] == 0.0)
    if np.any(empty):
        C[empty] = 1.0
        S[empty] = p
    P = C / S
    Y = np.zeros((n, p), dtype=int)
    for i in range(n):
        Y[i] = rng.multinomial(N[i], P[i])
    return Y, Z
