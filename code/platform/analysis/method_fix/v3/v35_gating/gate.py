"""gate.py — v3.5 门控模块（data-only，禁止读 truth）。

构造性 data-only 验证：`gate_diagnostics` / `gate_decision` 的全部输入为
  Y      : 观测计数矩阵 (n, p)            —— 数据
  N      : 每样本深度（offset 锚）        —— 数据
  group  : 分组向量 {0,1}                 —— 试验设计
  phi_hat, cnt_veto, w_hat : 估计器拟合输出（φ̂、撞界否决、逐样本-类群
           结构零后验权重 Ŵ；Ŵ 由拟合流程产出，本身不含真值）
  alpha  : DA 检验名义水平（默认 0.05）
模块内部不 import simulation_v3.generators、不读 truth 字典；
真值仅在 v35 验证脚本中用于事后评估门控表现（不进门控输入）。

三组件
  C1 可识别性：φ̂ 撞下/上界、veto、depth_med/(3φ̂)
  C2 可检性/校准：n、zero_rate、低流行率类群占比、est 臂 fallback 率、
     **置换校准率**（组标签 K 次置换，同掩码重跑 abs_nb_glm，零假设下
     拒绝率 >> α ⇒ 该 regime 检验器反保守 ⇒ 关门）
  C3 误差独立性预检：Ŵ ~ group / log-depth 相关（泄漏风险旗标）

回退语义（预注册）：关门格 = 声明型回退（不做 DA 断言、零拒绝）；
回退到 placeholder 不算受控（v3.4 实测 placeholder 在不可检层 FDR 0.342）。
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# ---- 候选门控规则（阈值在 LOCO 导出，默认值经 v3.4 网格验证后固化） --------
# 阈值经 v3.4 网格 LOCO（8 折，折间一致）固化，见 V35_MEMO §3。
DEFAULT_RULES = dict(
    perm_fdr_max=0.25,    # C2: est 臂置换 null-FDR（有拒绝的置换占比）上限
    tested_min=0.50,      # C2: est 臂 n_tested 占比下限
    phi_lower=0.06,       # C1: φ̂ 撞下界（数值容差内的下界邻域）
)


def leakage_precheck(w_hat: np.ndarray, group: np.ndarray,
                     N: np.ndarray) -> dict:
    """C3：零细胞内 Ŵ 与 group / log-depth 的相关（点二列 r + t 检验 p）。

    w_hat 为逐 (sample, taxon) 后验权重（Y>0 处按约定为 0，此处只用零细胞）。
    """
    group = np.asarray(group, dtype=float)
    w = np.asarray(w_hat, dtype=float)
    g = np.broadcast_to(group[:, None], w.shape)
    nd = np.broadcast_to(np.log(np.asarray(N, dtype=float))[:, None],
                         w.shape)
    zero = w > 0  # Ŵ 仅在 Y==0 处 >0（validate_weights 约定）
    out = dict(n_zero=int(zero.sum()))
    if zero.sum() < 20 or np.unique(g[zero]).size < 2:
        out.update(w_group_r=np.nan, w_group_p=np.nan,
                   w_depth_r=np.nan, w_depth_p=np.nan)
        return out
    r_g, p_g = stats.pearsonr(w[zero], g[zero])
    r_d, p_d = stats.pearsonr(w[zero], nd[zero])
    out.update(w_group_r=float(r_g), w_group_p=float(p_g),
               w_depth_r=float(r_d), w_depth_p=float(p_d))
    return out


def basic_diagnostics(Y: np.ndarray, N: np.ndarray, group: np.ndarray,
                      phi_hat: float, cnt_veto: bool, w_hat: np.ndarray,
                      est_n_tested_frac: float, phi_lower: float = 0.06
                      ) -> dict:
    """C1 + C2 的非置换部分（全部 data-only）。"""
    Y = np.asarray(Y, dtype=float)
    D = (Y > 0).astype(float)
    prev = D.mean(axis=0)
    depth_med = float(np.median(N))
    return dict(
        log_phi_hat=float(np.log(max(phi_hat, 1e-12))),
        phi_hat=float(phi_hat),
        phi_on_lower=bool(phi_hat <= phi_lower),
        phi_on_upper=bool(phi_hat >= 5e4),
        cnt_veto=bool(cnt_veto),
        depth_over_3phi=float(depth_med / (3.0 * max(phi_hat, 1e-12))),
        depth_med=depth_med,
        n_total=int(Y.shape[0]),
        zero_rate=float((Y == 0).mean()),
        prev_low_frac=float((prev < 0.1).mean()),
        w_hat_mean=float(w_hat[w_hat > 0].mean()) if (w_hat > 0).any() else 0.0,
        w_hat_mask_frac=float((w_hat >= 0.5).mean()),
        est_tested_frac=float(est_n_tested_frac),
    )


def gate_decision(diag: dict, rules: dict = DEFAULT_RULES) -> dict:
    """门控裁决：ON（做 DA 断言）/ OFF（声明型回退，零拒绝）+ 触发原因。"""
    reasons = []
    if diag["perm_fdr_est"] > rules["perm_fdr_max"]:
        reasons.append(f"R1_permFDR>{rules['perm_fdr_max']}")
    if diag["est_tested_frac"] < rules["tested_min"]:
        reasons.append(f"R2_tested<{rules['tested_min']}")
    if diag["phi_on_lower"]:
        reasons.append("R3_phi_lower")
    return dict(gate_on=len(reasons) == 0, reasons=";".join(reasons))


def permutation_calibration(Y, N, group, w_hat, abs_nb_glm, K=20,
                            alpha=0.05, seed=20260305):
    """C2 核心：组标签 K 次置换，est/placeholder 两臂各自重跑 abs_nb_glm，
    记录零假设拒绝率（n_rej/p 的跨置换均值）。

    abs_nb_glm 由调用方注入（避免本模块依赖 simulation_v3）。
    置换种子为固定常数（data-independent），跨 run 可比。
    """
    rng = np.random.default_rng(seed)
    group = np.asarray(group)
    p = Y.shape[1]
    keep = ~((Y == 0) & (w_hat >= 0.5))
    rates_est, rates_plac = [], []
    fdr_est, fdr_plac = [], []
    for _ in range(K):
        gperm = rng.permutation(group)
        r_e = abs_nb_glm(Y, gperm, N=N, W=keep.astype(float), alpha=alpha)
        r_p = abs_nb_glm(Y, gperm, N=N, W=None, alpha=alpha)
        rates_est.append(r_e["reject"].sum() / p)
        rates_plac.append(r_p["reject"].sum() / p)
        # 全局 null 下任何拒绝都是假阳性 ⇒ 单次置换 FDP = 1{有拒绝}
        fdr_est.append(float(r_e["reject"].sum() > 0))
        fdr_plac.append(float(r_p["reject"].sum() > 0))
    return dict(perm_rej_rate_est=float(np.mean(rates_est)),
                perm_rej_rate_plac=float(np.mean(rates_plac)),
                perm_fdr_est=float(np.mean(fdr_est)),
                perm_fdr_plac=float(np.mean(fdr_plac)),
                perm_K=int(K), perm_alpha=float(alpha))
