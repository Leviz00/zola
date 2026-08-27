"""model.py — 三层零分解模型的基线模拟器与 g 函数闭式。

权威来源（只读）：
  - outline_src/sections/sec3_framework.tex（三层模型、命题 prop:ident）
  - fixes/fix_N1.md（g 闭式、digamma 展开、可识别区域 e_j、精度合并为单一 φ）
  - fixes/fix_N3.md（ρ 交互参数化）
  - fixes/fix_N4.md（推断方案分工）

本模块实现【基线版】：ρ≡1、无协变量、无因子结构、单一精度 φ（测量层取
多项极限，全部过度离散由组成层承载，fix_N1 §3.1）。即

  存在层   Z_ij ~ Bernoulli(π_j)
  组成层   θ_i  ~ Dirichlet(φ · θ̄)                     （Σ_j θ̄_j = 1）
  测量层   p_i  = (θ_i ⊙ Z_i) / Σ_k θ_ik Z_ik,  Y_i ~ Multinomial(N_i, p_i)

在此基线下，【条件于其余类群在场】的逐类群边际恰为零膨胀 beta–二项（ZIBB）：
  P(Y_ij = 0 | N_i, Z_{i,-j}) = (1-π_j) + π_j · g(N_i; θ̄_j, φ),
  g(N;θ,φ) = E[(1-p)^N],  p ~ Beta(φθ, φ(1-θ)).

完整三层模型（因子结构 λ_i、协变量 W_i、检出效率 ρ_ij）留接口 stub，
见文件末尾 simulate_full / 说明。
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.integrate import quad

__all__ = [
    "log_g", "g_closed", "g_product", "g_quad", "g_gauss_jacobi",
    "g_monte_carlo",
    "S_N", "U_N", "one_minus_g_order1", "one_minus_g_order2",
    "dlogg_dtheta", "dlogg_dphi",
    "detection_prob", "effective_detection_strength",
    "simulate_three_layer", "simulate_zibb_marginal", "simulate_full",
]


# ---------------------------------------------------------------------------
# g 函数：闭式（Gamma 形式）、显式乘积、数值积分、Monte Carlo
# ---------------------------------------------------------------------------

def log_g(N, theta, phi):
    """log g(N;θ,φ) 的闭式（Gamma 形式，fix_N1 §1.1，数值稳定）。

    g(N;θ,φ) = Γ(φ(1-θ)+N) Γ(φ) / [ Γ(φ(1-θ)) Γ(φ+N) ].

    参数可广播：N 整数（≥0），θ∈(0,1)，φ>0。N=0 时 g=1（log g=0）。
    """
    # 不用 dtype=float 强转，以保留 complex（复步长数值微分需要）
    N = np.asarray(N) * 1.0
    theta = np.asarray(theta) * 1.0
    phi = np.asarray(phi) * 1.0
    a = phi * (1.0 - theta)  # Beta 的“缺席”形状参数
    if (np.iscomplexobj(N) or np.iscomplexobj(theta) or np.iscomplexobj(phi)):
        from scipy.special import loggamma as _lg  # gammaln 仅支持实数
    else:
        _lg = gammaln
    return (_lg(a + N) - _lg(a) + _lg(phi) - _lg(phi + N))


def g_closed(N, theta, phi):
    """g(N;θ,φ) 闭式值（exp of log_g）。"""
    return np.exp(log_g(N, theta, phi))


def g_product(N, theta, phi):
    """g 的显式上升阶乘乘积 ∏_{r=0}^{N-1} (φ(1-θ)+r)/(φ+r)。

    仅用于小 N 的交叉验证（O(N) 标量循环，不作生产用途）。
    """
    N = int(N)
    out = 1.0
    a = phi * (1.0 - theta)
    for r in range(N):
        out *= (a + r) / (phi + r)
    return out


def g_quad(N, theta, phi):
    """g 的数值积分定义：对 p~Beta(φθ, φ(1-θ)) 求 E[(1-p)^N]。

    用 scipy.integrate.quad 直接积分（不调用 scipy.stats.beta 的矩公式，
    以提供一条独立的第三方验证路径）。
    """
    from scipy.special import betaln
    a, b = phi * theta, phi * (1.0 - theta)
    # QUADPACK QAWSE：端点代数奇异权 p^{a-1}(1-p)^{b-1}（恰为未归一 Beta 密
    # 度），被积函数 (1-p)^N 光滑，高精度可达。
    val, _ = quad(lambda p: (1.0 - p) ** N, 0.0, 1.0,
                  weight="alg", wvar=(a - 1.0, b - 1.0),
                  epsabs=1e-13, epsrel=1e-13, limit=200)
    return val * np.exp(-betaln(a, b))


def g_gauss_jacobi(N, theta, phi, n_nodes=None):
    """g 的 Gauss–Jacobi 求积（第四条验证路径，限中小 N）。

    p~Beta(a,b)，a=φθ，b=φ(1−θ)：E[f(p)] = 2^{-(a+b-1)}/B(a,b)
    · Σ_i w_i f((1+x_i)/2)，(x_i,w_i) 为权 (1-x)^{b-1}(1+x)^{a-1} 的
    Gauss–Jacobi 节点。f(p)=(1-p)^N 为 N 次多项式，n_nodes > N/2 时
    代数上精确；但 float64 节点计算在大 m 下损失精度（N≳200 时误差
    ~1e-7），故仅作中小 N 的辅助对照，正式三方对照用 g_quad。
    """
    from scipy.special import roots_jacobi, betaln
    a, b = phi * theta, phi * (1.0 - theta)
    m = int(n_nodes or max(64, N + 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        # scipy 内部在 k=1 处用 np.where 规避 0/0（φ=1 时触发），警告无害
        x, w = roots_jacobi(m, b - 1.0, a - 1.0)
    f = ((1.0 - x) / 2.0) ** N
    return float(2.0 ** (-(a + b - 1.0)) * np.exp(-betaln(a, b))
                 * np.dot(w, f))


def g_monte_carlo(N, theta, phi, n_draws=400_000, seed=42):
    """g 的 Monte Carlo 估计（p~Beta(φθ,φ(1-θ))，E[(1-p)^N]）。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(phi * theta, phi * (1.0 - theta), size=n_draws)
    return float(np.mean((1.0 - p) ** N))


# ---------------------------------------------------------------------------
# digamma / trigamma 渐近展开（fix_N1 §1.2）
# ---------------------------------------------------------------------------

def S_N(N, phi):
    """S_N(φ) = Σ_{r=0}^{N-1} 1/(φ+r) = ψ(φ+N) − ψ(φ)。"""
    return digamma(np.asarray(phi, dtype=float) + np.asarray(N, dtype=float)) \
        - digamma(np.asarray(phi, dtype=float))


def U_N(N, phi):
    """U_N(φ) = Σ_{r=0}^{N-1} 1/(φ+r)^2 = ψ'(φ) − ψ'(φ+N)。"""
    return polygamma(1, np.asarray(phi, dtype=float)) \
        - polygamma(1, np.asarray(phi, dtype=float) + np.asarray(N, dtype=float))


def one_minus_g_order1(N, theta, phi):
    """一阶展开 1−g ≈ φθ·S_N(φ)（θ→0，φ,N 固定）。"""
    return phi * theta * S_N(N, phi)


def one_minus_g_order2(N, theta, phi):
    """二阶展开 1−g ≈ 1 − exp(−φθ·S_N(φ) − (φθ)²/2·U_N(φ))。"""
    logg = -phi * theta * S_N(N, phi) - 0.5 * (phi * theta) ** 2 * U_N(N, phi)
    return 1.0 - np.exp(logg)


# ---------------------------------------------------------------------------
# log g 对 (θ, φ) 的解析导数（复合似然得分函数所需）
# ---------------------------------------------------------------------------

def dlogg_dtheta(N, theta, phi):
    """∂ log g / ∂θ = −φ [ ψ(φ(1−θ)+N) − ψ(φ(1−θ)) ]。"""
    a = phi * (1.0 - theta)
    return -phi * (digamma(a + N) - digamma(a))


def dlogg_dphi(N, theta, phi):
    """∂ log g / ∂φ = (1−θ)[ψ(φ(1−θ)+N) − ψ(φ(1−θ))] − [ψ(φ+N) − ψ(φ)]。

    直接对 Gamma 闭式求导。
    """
    a = phi * (1.0 - theta)
    return ((1.0 - theta) * (digamma(a + N) - digamma(a))
            - (digamma(phi + N) - digamma(phi)))


# ---------------------------------------------------------------------------
# 检出曲线与有效一阶检出强度（命题 prop:ident 与其 remark）
# ---------------------------------------------------------------------------

def detection_prob(N, pi, theta, phi):
    """检出曲线 D(N) = P(Y_ij>0 | N) = π_j · [1 − g(N; θ̄_j, φ)]。"""
    return pi * (1.0 - g_closed(N, theta, phi))


def effective_detection_strength(theta, phi, N_min):
    """有效一阶检出强度 e_j = φθ̄ [ψ(φ+N_min) − ψ(φ)]。

    fix_N1 remark（可识别曲线图）：控制可识别性的是 e_j 而非 θ̄·N_min。
    大 N 区近似 φθ̄·log(1+N_min/φ)。
    """
    return phi * theta * S_N(N_min, phi)


# ---------------------------------------------------------------------------
# 模拟生成器
# ---------------------------------------------------------------------------

def simulate_three_layer(pi, theta_bar, phi, N, rng):
    """基线三层生成器（ρ≡1、无协变量、无因子结构、单一精度 φ）。

    参数
    ----
    pi        : (p,) 存在概率 π_j
    theta_bar : (p,) 基线组成 θ̄，须 Σ_j θ̄_j = 1
    phi       : 标量，合并后的单一精度（组成层总浓度 α0，测量层取多项极限）
    N         : (n,) 正整数测序深度
    rng       : numpy Generator

    返回
    ----
    Y : (n, p) int 计数矩阵；Z : (n, p) bool 存在指示
    """
    pi = np.asarray(pi, dtype=float)
    theta_bar = np.asarray(theta_bar, dtype=float)
    N = np.asarray(N, dtype=int)
    assert np.isclose(theta_bar.sum(), 1.0), "theta_bar 须归一"
    n = N.shape[0]
    p = pi.shape[0]

    # 存在层
    Z = rng.random((n, p)) < pi[None, :]
    # 组成层（每样本独立 Dirichlet，精度 φ）
    Theta = rng.dirichlet(phi * theta_bar, size=n)  # (n, p)
    # 屏蔽 + 重新归一（式 renorm）
    C = Theta * Z
    S = C.sum(axis=1, keepdims=True)
    # 所有类群缺席的样本（概率可忽略）防御性处理：退化为均匀
    empty = (S[:, 0] == 0.0)
    if np.any(empty):
        C[empty] = 1.0
        S[empty] = p
    P = C / S
    # 测量层：多项（DM 精度 → ∞ 极限）
    Y = np.zeros((n, p), dtype=int)
    for i in range(n):
        Y[i] = rng.multinomial(N[i], P[i])
    return Y, Z


def simulate_zibb_marginal(pi, theta_bar, phi, N, rng):
    """逐类群精确 ZIBB 边际生成器（类群间独立）。

    对应三层模型【条件于其余类群在场】的严格逐类群边际（fix_N1 §3.1
    条件 1–3 全部成立的情形）：Z_ij~Bern(π_j)，
    p_ij|Z_ij=1 ~ Beta(φθ̄_j, φ(1−θ̄_j))，Y_ij|Z_ij=1 ~ Binomial(N_i, p_ij)。

    用途：复合似然估计器在该生成机制下是精确 MLE（无归一化耦合误差 E1），
    作为自校准的对照；三层联合生成器下的偏差即 E1 的量化。
    注意：此生成器不要求 Σθ̄=1（逐类群独立），θ̄_j 直接是 Beta 均值。
    """
    pi = np.asarray(pi, dtype=float)
    theta_bar = np.asarray(theta_bar, dtype=float)
    N = np.asarray(N, dtype=int)
    n, p = N.shape[0], pi.shape[0]
    Z = rng.random((n, p)) < pi[None, :]
    P = rng.beta(phi * theta_bar[None, :],
                 phi * (1.0 - theta_bar[None, :]), size=(n, p))
    Y = rng.binomial(N[:, None], P * Z)
    return Y, Z


def simulate_full(*args, **kwargs):
    """【接口 stub】完整三层生成器：因子结构 λ_i、协变量 γ_j^T W_i、
    类群×批次检出效率 ρ_ij（logit ρ_ij = δ_j + δ_{j,B_i}，fix_N3 §2）。

    当前版本不实现（见 README「已知局限」）。计划接口：
      simulate_full(gamma, U, W, batch, delta, delta_inter, theta_bar,
                    phi, N, rng)
    其中存在层 logit π_ij = γ_j^T W_i + u_j^T λ_i，λ_i~N(0,I_r)。
    """
    raise NotImplementedError(
        "完整三层模型（因子结构/协变量/ρ 检出效率层）留作扩展，"
        "见 README 已知局限；ρ 参数化遵循 fix_N3 的类群×批次交互。"
    )
