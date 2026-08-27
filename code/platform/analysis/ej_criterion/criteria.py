"""criteria.py — e_j 有限样本可识别判据的三个候选形式化（M4 修复）。

权威背景：
  - 理论横轴 e_j = φ θ̄_j [ψ(φ+N_min)−ψ(φ)]（outline sec3 remark, fix_N1）。
  - 真实数据教训（realdata/REPORT.md §6 差异 3）：e_j(N_min) 对深度离群值
    退化、不含 n 与深度分布、跨数据集不可比。

三个候选判据（全部以逐类群检出指示 Bernoulli 复合似然为推断模型）：

(a) SE 判据（现行真实数据操作的规范化）
    类群 j 可识别 ⟺ SE_G(logit π̂_j) < c_a 且未撞参数边界。
    SE_G² = [A⁻¹ B A⁻¹]_{αα}（Godambe 三明治，A=工作模型期望 Fisher，
    B=Σ_i u_i u_iᵀ）。等价地 Ĩ_j := 1/SE_G² > 1/c_a²，即"经验 Godambe
    信息超过阈值"。SE<1 ⟺ logit π 的 95% CI 半宽 <2（π̂ 不被检出噪声淹没）。
    注意：SE_G 是事后量（需要拟合），含 B≠A 的耦合/误设修正。

(b) 稳健深度版 e_j（修复 N_min 离群退化，但仍不含 n）
    (b1) 分位版：e_j^(q10) = φ θ̄_j [ψ(φ+N_q10)−ψ(φ)]，N_q10 为深度的
         10% 分位数（替换 N_min）。
    (b2) 期望版：ē_j = φ θ̄_j · E_N[S_N(φ)] = φ θ̄_j · (1/n)Σ_i S_{N_i}(φ)，
         即"平均每样本有效一阶检出强度"（检出曲线小 θ 展开系数的样本均值）。
    阈值 t_b 需校准；预期跨数据集不可移植（缺 n）。

(c) 信息判据（推荐候选：理论横轴的有限样本化）
    逐类群（α_j, β_j）=(logit π_j, log θ̄_j) 的工作模型 Fisher（φ 已知或
    剖面掉）：
      q_i = π(1−g_i),  dq/dα = (1−g_i)π(1−π),  dq/dβ = −π g_i ∂log g_i/∂θ·θ,
      w_i = 1/(q_i(1−q_i)),
      A = Σ_i w_i v_i v_iᵀ (2×2),  v_i = (dq/dα, dq/dβ)。
    剖面 Fisher 信息
      I_j = A_{αα} − A_{αβ}²/A_{ββ} = 1/[A⁻¹]_{αα},
    判据：I_j ≥ t_c（t_c=1 ⟺ 理论 SE ≤ 1），实测数据附"未撞边界"条件。
    I_j 与 e_j 的解析关系（θ̄→0 一阶展开，g_i≈1，∂log g/∂θ≈−φ S_{N_i}）：
      A_{αα} ≈ π(1−π)² φθ̄ Σ_i S_{N_i} = π(1−π)² · n · ē_j,
      I_j    = A_{αα}(1−ρ_{αβ}²),
    其中 ρ_{αβ}=A_{αβ}/√(A_{αα}A_{ββ}) 为脊方向相关（θ̄→0 时 |ρ|→1，
    I_j→0，精确计算自动捕获该塌缩）。故
      log I_j ≈ log n + log ē_j + log[π(1−π)²(1−ρ_{αβ}²)]：
    I_j 自然包含样本量 n、完整深度分布（经 Σ_i S_{N_i}）与脊收缩因子，
    是 e_j 的正确有限样本推广；e_j 仅是 I_j/n 在单样本、无线性区外的
    一阶极限。设计用途：给定目标 I_j≥t_c 反解所需 (n, 深度分布)。
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import log_g, dlogg_dtheta, S_N  # noqa: E402

_EPS = 1e-12


# ---------------------------------------------------------------------------
# (b) 稳健深度版 e_j
# ---------------------------------------------------------------------------

def ej_min(theta, phi, depths):
    """原版 e_j（N_min）：φ θ̄ [ψ(φ+N_min)−ψ(φ)]。"""
    return phi * theta * S_N(np.min(depths), phi)


def ej_quantile(theta, phi, depths, q=0.10):
    """(b1) 分位版 e_j^(q)：N_min 替换为深度 q 分位数（默认 q=0.10）。"""
    return phi * theta * S_N(np.quantile(depths, q), phi)


def ej_mean(theta, phi, depths):
    """(b2) 期望版 ē_j = φ θ̄ · (1/n) Σ_i S_{N_i}(φ)（平均每样本强度）。"""
    return phi * theta * float(np.mean(S_N(np.asarray(depths, float), phi)))


# ---------------------------------------------------------------------------
# (c) 逐类群剖面 Fisher 信息（φ 已知/剖面；理论量，仅需参数与深度分布）
# ---------------------------------------------------------------------------

def fisher_per_taxon(pi, theta, phi, depths):
    """逐类群 (α,β)=(logit π, log θ) 的 2×2 期望 Fisher（工作模型）。

    返回 (A, I_profiled, rho_ab, se_alpha_theory)：
      A          : (2,2) 矩阵 Σ_i w_i v_i v_iᵀ
      I_profiled : A_{αα} − A_{αβ}²/A_{ββ} = 1/[A⁻¹]_{αα}
      rho_ab     : A_{αβ}/√(A_{αα}A_{ββ})（脊方向相关）
      se_alpha_theory : 1/√I_profiled（正确设定下 SE_G 的理论值）
    """
    N = np.asarray(depths, dtype=float)
    lg = log_g(N, theta, phi)
    g = np.exp(lg)
    q = np.clip(pi * (1.0 - g), _EPS, 1.0 - _EPS)
    dq_da = (1.0 - g) * pi * (1.0 - pi)                  # ∂q/∂α
    dq_db = -pi * g * dlogg_dtheta(N, theta, phi) * theta  # ∂q/∂β
    w = 1.0 / (q * (1.0 - q))
    Aaa = float(np.sum(w * dq_da * dq_da))
    Aab = float(np.sum(w * dq_da * dq_db))
    Abb = float(np.sum(w * dq_db * dq_db))
    A = np.array([[Aaa, Aab], [Aab, Abb]])
    I_prof = Aaa - Aab * Aab / max(Abb, _EPS)
    rho = Aab / np.sqrt(max(Aaa * Abb, _EPS))
    se = 1.0 / np.sqrt(max(I_prof, _EPS))
    return A, I_prof, rho, se


def info_criterion(pi, theta, phi, depths, threshold=1.0):
    """(c) 信息判据：I_j ≥ threshold（默认 1 ⟺ 理论 SE(logit π) ≤ 1）。"""
    _, I_prof, _, _ = fisher_per_taxon(pi, theta, phi, depths)
    return I_prof >= threshold


# ---------------------------------------------------------------------------
# (a) SE 判据的规范化封装（经验版：用拟合输出的 Godambe SE）
# ---------------------------------------------------------------------------

def se_criterion(se_logit_pi, on_boundary, cutoff=1.0):
    """(a) 可识别 ⟺ SE_G(logit π̂) < cutoff 且未撞边界（现行 REPORT 判据）。"""
    return (se_logit_pi < cutoff) & (~on_boundary)


def n_scaled_ej(theta, phi, depths):
    """辅助量：n·ē_j（期望总检出强度，≈ 稀有区期望检出计数/π 的一阶近似）。"""
    return len(np.asarray(depths)) * ej_mean(theta, phi, depths)
